"""Frozen L0 wire contracts for operator matrix, bench plans, env fingerprints & campaigns.

[Role] Define 100% strictly typed, immutable L0 schema contracts shared across
  ``mindie-sim`` and ``mindie-profiler`` via four canonical JSON/JSONL wire
  schemas (``mindie-bench-plan/2``, ``mindie-bench-env/1``,
  ``mindie-bench-observation/2``, ``mindie-bench-campaign/1``) plus the
  outward-facing ``OperatorMatrixManifest`` / ``HardwareNominalSpec`` /
  ``StepSummary`` / ``HbmAdmissibilityReceipt`` deliverable records.
[Invariants]
  * Pure L0 module: zero runtime imports of ``mindie_profiler`` or L1–L5
    packages; deterministic canonical JSON serialization (``sort_keys=True``,
    ``separators=(",", ":")``, ``ensure_ascii=True``, ``allow_nan=False``).
  * ``CanonicalKernelKey`` fixes all 31 physical execution axes (incl. strides,
    storage_offsets, KV allocator/kernel block sizes, CP interleave, page
    geometry, per-expert/per-rank token counts, ACLGraph padded bucket,
    workspace policy, stream dependencies, residency, and multi-node topology
    axes) and strictly forbids model/layer/split/run/prediction leakage keys.
  * ``HardwareNominalSpec`` stores un-derated public datasheet peaks from
    ``ascend.json`` (``utilization == 1.0``, ``launch_us == 0.0``), NEVER
    ``PrimitiveTable`` sustained rates or link framing deratings.
  * ``OperatorMatrixRow`` separates ``hbm_read_bytes`` and ``hbm_write_bytes``
    and reports both ``mbu_rw_pct`` and ``mbu_read_pct``; ``StepSummary``
    reports both algorithmic ``model_mbu_pct`` and physical ``hbm_bus_util_pct``.
  * ``HbmAdmissibilityReceipt`` enforces dual-peak admission (``runtime_peak``
    AND ``load_peak`` per die plus optional HiSparse host DDR capacity).
[Timeouts & Locks] Pure in-memory validation and SHA-256 hashing; no I/O or locks.
[Side-Effect Policy] Strictly frozen dataclasses and read-only views; zero mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from .abi import (
    HARDWARE_USABLE_ABI_BACKENDS,
    SOC_ARCHS,
    SocArch,
)
from .common import ModelingError

BENCH_PLAN_V2_SCHEMA: Literal["mindie-bench-plan/2"] = "mindie-bench-plan/2"
BENCH_ENV_V1_SCHEMA: Literal["mindie-bench-env/1"] = "mindie-bench-env/1"
BENCH_OBSERVATION_V2_SCHEMA: Literal["mindie-bench-observation/2"] = "mindie-bench-observation/2"
BENCH_CAMPAIGN_V1_SCHEMA: Literal["mindie-bench-campaign/1"] = "mindie-bench-campaign/1"
OPERATOR_MATRIX_SCHEMA: Literal["mindie-sim.operator-matrix/1"] = "mindie-sim.operator-matrix/1"

CANONICAL_KEY_SIG_VERSION = 2
CANONICAL_DIGEST_HEX_LEN = 16
ENV_DIGEST_HEX_LEN = 16
SHA256_HEX_LEN = 64
CASE_ID_PREFIX = "k2-"
DEFAULT_SEQ_MIN = 512
DEFAULT_SEQ_MAX = 409600
DEFAULT_ANCHOR_LADDER: tuple[int, ...] = (512, 2048, 8192, 32768, 131072, 262144, 409600)
DEFAULT_CLIFF_BUDGET_PER_ARM = 8
BYTES_PER_GB = 1_000_000_000.0
US_PER_SECOND = 1_000_000.0
PCT_SCALE = 100.0
NOMINAL_UTILIZATION = 1.0
NOMINAL_LAUNCH_US = 0.0
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_16_RE = re.compile(r"^[0-9a-f]{16}$")
_SLUG_RE = re.compile(r"[^a-z0-9_]+")
_CANN_SLUG_RE = re.compile(r"[^a-z0-9._-]+")

KernelKeyKind: TypeAlias = Literal["operator", "subgraph", "collective"]
CaptureMode: TypeAlias = Literal["eager", "aclgraph"]
ResidencyMode: TypeAlias = Literal["cold_hbm", "warm_l2"]
CachePartition: TypeAlias = Literal["shared", "train", "heldout"]
ClockOperator: TypeAlias = Literal["busy_union", "elapsed_span"]
PlanSplit: TypeAlias = Literal["train", "heldout", "abi_rejection", "matrix"]
CliffKind: TypeAlias = Literal[
    "anchor", "wave_cliff", "ub_cliff", "l2_cliff", "kv_block_cliff", "index_pool_cliff",
]
ModelScopeKind: TypeAlias = Literal["target", "draft"]
PhaseKind: TypeAlias = Literal["prefill", "decode"]
AcceleratorCore: TypeAlias = Literal[
    "AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "HCCL", "DSA",
]
BoundType: TypeAlias = Literal[
    "Compute Bound (Cube)",
    "Compute Bound (Vector)",
    "Memory Bound (HBM)",
    "Communication Bound",
    "Latency Bound",
]
CampaignCaseStatus: TypeAlias = Literal[
    "ok", "abi_rejected", "abi_unsupported", "over_budget", "failed", "unsupported", "blocked",
]

VALID_ACCELERATOR_CORES: frozenset[str] = frozenset({
    "AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "HCCL", "DSA",
})
VALID_BOUND_TYPES: frozenset[str] = frozenset({
    "Compute Bound (Cube)",
    "Compute Bound (Vector)",
    "Memory Bound (HBM)",
    "Communication Bound",
    "Latency Bound",
})

OUTWARD_FORBIDDEN_INTERNAL_TOKENS: tuple[str, ...] = (
    "mindie_sim",
    "mindie-sim",
    "mindie_profiler",
    "Stage 0",
    "Stage 1",
    "Stage 2",
    "Stage 3",
    "L*",
    "l_star",
    "dual_cert",
    "Farkas",
    "OpOccurrence",
    "LayerRunProgram",
    "PlannedRecipe",
    "PrimitiveTable",
    "P01_",
    "P06_",
    "P18_",
    "arch22",
    "arch35",
    "cohort",
    "rep_dedup",
    "unbound_formula",
    "cell_id",
    "k2-",
    "sig-",
)

VALID_KEY_KINDS: frozenset[str] = frozenset({"operator", "subgraph", "collective"})
VALID_CAPTURE_MODES: frozenset[str] = frozenset({"eager", "aclgraph"})
VALID_RESIDENCY_MODES: frozenset[str] = frozenset({"cold_hbm", "warm_l2"})
VALID_CACHE_PARTITIONS: frozenset[str] = frozenset({"shared", "train", "heldout"})
VALID_CLOCK_OPERATORS: frozenset[str] = frozenset({"busy_union", "elapsed_span"})
VALID_PLAN_SPLITS: frozenset[str] = frozenset({"train", "heldout", "abi_rejection", "matrix"})
VALID_CLIFF_KINDS: frozenset[str] = frozenset({
    "anchor", "wave_cliff", "ub_cliff", "l2_cliff", "kv_block_cliff", "index_pool_cliff",
})
VALID_MODEL_SCOPES: frozenset[str] = frozenset({"target", "draft"})
VALID_PHASES: frozenset[str] = frozenset({"prefill", "decode"})
VALID_CASE_STATUSES: frozenset[str] = frozenset({
    "ok", "abi_rejected", "abi_unsupported", "over_budget", "failed", "unsupported", "blocked",
})

FORBIDDEN_PREDICTION_KEYS: frozenset[str] = frozenset({
    "predicted_ns",
    "predicted_ms",
    "predicted_us",
    "predicted_s",
    "predicted_duration",
    "sim_reference",
})
FORBIDDEN_CANONICAL_KEY_FIELDS: frozenset[str] = frozenset({
    "model_id",
    "layer_idx",
    "split",
    "run_id",
    "seq_len_label",
    "concurrency_label",
    "plan_id",
    "case_id",
}) | FORBIDDEN_PREDICTION_KEYS

CANONICAL_KERNEL_KEY_FIELDS: tuple[str, ...] = (
    "sig_version",
    "kind",
    "sku",
    "soc_arch",
    "abi_backend",
    "upstream_abi",
    "bench_builder",
    "canonical_shapes",
    "strides",
    "storage_offsets",
    "dtypes",
    "layouts",
    "kv_allocator_block_size",
    "kv_kernel_block_size",
    "cp_kv_cache_interleave_size",
    "page_geometry",
    "per_expert_token_counts",
    "per_rank_token_counts",
    "capture_mode",
    "aclgraph_padded_bucket",
    "workspace_policy",
    "stream_dependencies",
    "residency",
    "quant_meta",
    "op_attrs",
    "ordered_rank_placement",
    "link_tiers",
    "collective_algorithm",
    "split_sizes",
    "timed_nodes",
    "edges",
)

STATIC_ENV_BUCKET_FIELDS: tuple[str, ...] = (
    "sku",
    "soc_arch",
    "hw_geometry",
    "driver_version",
    "cann_version",
    "cann_innerversion",
    "cann_packages",
    "torch_version",
    "torch_npu_version",
    "va_commit",
    "triton_version",
    "custom_op_libs",
    "profiler_version",
    "timing_protocol_id",
    "cache_partition",
)

#: Packaged fabric collective op -> (builder_id, certified_upstream_abi) map.
FABRIC_COLLECTIVE_ABI_MAP: Mapping[str, tuple[str, str]] = MappingProxyType({
    "all_reduce": ("all_reduce", "hcclAllReduce"),
    "allreduce": ("all_reduce", "hcclAllReduce"),
    "quantized_all_reduce": ("quantized_all_reduce", "hcclAllReduceInt8"),
    "all_gather": ("all_gather", "hcclAllGather"),
    "allgather": ("all_gather", "hcclAllGather"),
    "reduce_scatter": ("reduce_scatter", "hcclReduceScatter"),
    "all_to_all": ("all_to_all", "hcclAlltoAll"),
    "alltoall": ("all_to_all", "hcclAlltoAll"),
    "all_to_all_v": ("all_to_all_v", "hcclAlltoAllV"),
    "alltoallv": ("all_to_all_v", "hcclAlltoAllV"),
    "p2p_send_recv": ("p2p_send_recv", "hcclSendRecv"),
    "send_recv": ("p2p_send_recv", "hcclSendRecv"),
    "pp_p2p": ("p2p_send_recv", "hcclSendRecv"),
})

_DTYPE_PEAK_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "bf16": ("bf16", "fp16"),
    "bfloat16": ("bf16", "fp16"),
    "fp16": ("fp16", "bf16"),
    "float16": ("fp16", "bf16"),
    "fp32": ("fp32",),
    "float32": ("fp32",),
    "hf32": ("hf32", "fp32"),
    "int8": ("int8",),
    "w8a8": ("int8",),
    "int4": ("int4",),
    "w4a8": ("int8", "int4"),
    "fp8": ("fp8_e4m3", "fp8_e5m2"),
    "fp8_e4m3": ("fp8_e4m3", "fp8_e5m2"),
    "fp8_e4m3fn": ("fp8_e4m3", "fp8_e5m2"),
    "fp8_e5m2": ("fp8_e5m2", "fp8_e4m3"),
    "hif8": ("hifloat8", "fp8_e4m3"),
    "hifloat8": ("hifloat8", "fp8_e4m3"),
    "mxfp8": ("mxfp8", "fp8_e4m3"),
    "fp4": ("fp4_e2m1", "mxfp4"),
    "fp4_e2m1": ("fp4_e2m1", "mxfp4"),
    "mxfp4": ("mxfp4", "fp4_e2m1"),
})

_LINK_TIER_CHANNEL_MAP: Mapping[str, str] = MappingProxyType({
    "d2d_intra_package": "d2d_intra_package",
    "intra_package": "d2d_intra_package",
    "sio": "d2d_intra_package",
    "d2d_intra_node": "d2d_intra_node",
    "intra_node": "d2d_intra_node",
    "intra_node_hccs": "d2d_intra_node",
    "intra_4p": "d2d_intra_node",
    "intra_8p": "d2d_intra_node",
    "hccs": "d2d_intra_node",
    "d2d_inter_node": "d2d_inter_node",
    "inter_node": "d2d_inter_node",
    "inter_node_roce": "d2d_inter_node",
    "inter_pod": "d2d_inter_node",
    "roce": "d2d_inter_node",
    "h2d": "h2d",
    "d2h": "d2h",
    "rh2d": "rh2d",
    "rd2h": "rd2h",
})

_BENCH_PLAN_KNOWN_FIELDS: frozenset[str] = frozenset({
    "schema",
    "plan_id",
    "plan_kind",
    "sku",
    "soc_arch",
    "stage",
    "cache_partition",
    "world_size",
    "training_manifest_sha256",
    "cases",
    "reverse_index",
    "plan_sha256",
})

_CAMPAIGN_CASE_KNOWN_FIELDS: frozenset[str] = frozenset({
    "acquisition_uid",
    "case_id",
    "canonical_key_digest",
    "boundary",
    "split",
    "status",
    "qualified",
    "timing_mode",
    "capture_mode",
    "clock_operator",
    "residency",
    "rank",
    "participating_ranks",
    "timed_nodes",
    "p50_us",
    "p90_us",
    "p99_us",
    "min_us",
    "iqr_us",
    "cv_pct",
    "samples_count",
    "observed_clock_mhz",
    "hbm_peak_bytes",
    "actual_executed_shape",
    "cache_status",
    "measured_in_run_id",
    "run_shard",
    "cold_coverage_verified",
    "traced_interval_verified",
    "family",
    "geometry_key",
    "upstream_abi",
    "abi_backend",
    "bench_builder",
    "capture_uri",
    "raw_sha256",
})

_CAMPAIGN_KNOWN_FIELDS: frozenset[str] = frozenset({
    "schema",
    "campaign_id",
    "env_fingerprint_id",
    "cache_partition",
    "plan_id",
    "plan_sha256",
    "version_fingerprint",
    "clock_operator",
    "observations_jsonl",
    "observations_sha256",
    "run_shards",
    "training_manifest_sha256",
    "cases",
    "blocked_cases",
    "worker_diagnostics",
    "campaign_sha256",
})

_CHILD_KERNEL_KEY_MARKERS: frozenset[str] = frozenset({"kind", "sku", "soc_arch", "bench_builder"})


def resolve_fabric_collective_abi(collective_op: str) -> tuple[str, str]:
    """Resolve ``(builder_id, upstream_abi)`` for a fabric collective without tools/."""
    key = str(collective_op or "").strip().lower()
    entry = FABRIC_COLLECTIVE_ABI_MAP.get(key)
    if entry is None:
        raise ModelingError(f"fabric_collective_unknown:{collective_op}")
    return entry


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelingError(f"non_finite_json_float:{value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain_json(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    raise ModelingError(f"unsupported_canonical_json_type:{type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelingError(f"non_finite_json_float:{value!r}")
        return value
    if isinstance(value, Mapping):
        frozen = {str(k): _freeze_json(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ModelingError(f"unsupported_canonical_json_type:{type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize to deterministic ASCII canonical JSON without trailing newline."""
    plain = _plain_json(value)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    """Return full 64-hex SHA-256 of canonical JSON encoding."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def assert_no_forbidden_prediction_keys(value: Any, path: str = "$") -> None:
    """Recursively reject caller-supplied ``predicted_*`` or ``sim_reference`` keys."""
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            low = key.lower()
            if low in FORBIDDEN_PREDICTION_KEYS or low.startswith("predicted_"):
                raise ModelingError(f"forbidden_prediction_key:{path}.{key}")
            assert_no_forbidden_prediction_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            assert_no_forbidden_prediction_keys(item, f"{path}[{idx}]")


def assert_no_outward_internal_tokens(text: str, *, context: str = "outward_bundle") -> None:
    """Fail closed if any internal simulator/profiler jargon leaks into outward deliverables."""
    for token in OUTWARD_FORBIDDEN_INTERNAL_TOKENS:
        if token in text:
            raise ModelingError(f"outward_internal_token_leak:{context}:{token}")


def _req_nonempty_str(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _req_int(value: Any, code: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingError(f"{code}:{value!r}")
    if minimum is not None and value < minimum:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _opt_int(value: Any, code: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _req_int(value, code, minimum=minimum)


def _req_float(
    value: Any, code: str, *, minimum: float | None = None, strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"{code}:{value!r}")
    num = float(value)
    if not math.isfinite(num):
        raise ModelingError(f"{code}:{value!r}")
    if strictly_positive and num <= 0.0:
        raise ModelingError(f"{code}:{value!r}")
    if minimum is not None and num < minimum:
        raise ModelingError(f"{code}:{value!r}")
    return num


def _opt_float(
    value: Any, code: str, *, minimum: float | None = None, strictly_positive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _req_float(value, code, minimum=minimum, strictly_positive=strictly_positive)


def _validate_sha256_hex(value: Any, code: str) -> str:
    text = _req_nonempty_str(value, code)
    if not _SHA256_HEX_RE.fullmatch(text):
        raise ModelingError(f"{code}:{text!r}")
    return text


def _validate_digest16(value: Any, code: str) -> str:
    text = _req_nonempty_str(value, code)
    if not _DIGEST_16_RE.fullmatch(text):
        raise ModelingError(f"{code}:{text!r}")
    return text


def _req_soc_arch(value: Any, code: str) -> SocArch:
    text = _req_nonempty_str(value, code).upper()
    if text == "A2":
        return "A2"
    if text == "A3":
        return "A3"
    if text == "A5":
        return "A5"
    raise ModelingError(f"{code}:{value!r}")


def _req_key_kind(value: Any) -> KernelKeyKind:
    text = _req_nonempty_str(value, "canonical_key_kind_required").lower()
    if text == "operator":
        return "operator"
    if text == "subgraph":
        return "subgraph"
    if text == "collective":
        return "collective"
    raise ModelingError(f"canonical_key_kind_invalid:{value!r}")


def _req_capture_mode(value: Any) -> CaptureMode:
    text = _req_nonempty_str(value, "capture_mode_required").lower()
    if text == "eager":
        return "eager"
    if text == "aclgraph":
        return "aclgraph"
    raise ModelingError(f"capture_mode_invalid:{value!r}")


def _req_residency(value: Any) -> ResidencyMode:
    text = _req_nonempty_str(value, "residency_required").lower()
    if text == "cold_hbm":
        return "cold_hbm"
    if text == "warm_l2":
        return "warm_l2"
    raise ModelingError(f"residency_invalid:{value!r}")


def _req_cache_partition(value: Any) -> CachePartition:
    text = _req_nonempty_str(value, "cache_partition_required").lower()
    if text == "shared":
        return "shared"
    if text == "train":
        return "train"
    if text == "heldout":
        return "heldout"
    raise ModelingError(f"cache_partition_invalid:{value!r}")


def _req_clock_operator(value: Any) -> ClockOperator:
    text = _req_nonempty_str(value, "clock_operator_required").lower()
    if text == "busy_union":
        return "busy_union"
    if text == "elapsed_span":
        return "elapsed_span"
    raise ModelingError(f"clock_operator_invalid:{value!r}")


def _req_plan_split(value: Any) -> PlanSplit:
    text = _req_nonempty_str(value, "plan_split_required").lower()
    if text == "train":
        return "train"
    if text == "heldout":
        return "heldout"
    if text == "abi_rejection":
        return "abi_rejection"
    if text == "matrix":
        return "matrix"
    raise ModelingError(f"plan_split_invalid:{value!r}")


def _req_cliff_kind(value: Any) -> CliffKind:
    text = _req_nonempty_str(value, "cliff_kind_required").lower()
    if text == "anchor":
        return "anchor"
    if text == "wave_cliff":
        return "wave_cliff"
    if text == "ub_cliff":
        return "ub_cliff"
    if text == "l2_cliff":
        return "l2_cliff"
    if text == "kv_block_cliff":
        return "kv_block_cliff"
    if text == "index_pool_cliff":
        return "index_pool_cliff"
    raise ModelingError(f"cliff_kind_invalid:{value!r}")


def _req_model_scope(value: Any) -> ModelScopeKind:
    text = _req_nonempty_str(value, "model_scope_required").lower()
    if text == "target":
        return "target"
    if text == "draft":
        return "draft"
    raise ModelingError(f"model_scope_invalid:{value!r}")


def _req_phase(value: Any) -> PhaseKind:
    text = _req_nonempty_str(value, "phase_required").lower()
    if text == "prefill":
        return "prefill"
    if text == "decode":
        return "decode"
    raise ModelingError(f"phase_invalid:{value!r}")


def _req_accelerator_core(value: Any) -> AcceleratorCore:
    text = _req_nonempty_str(value, "accelerator_core_required").upper()
    if text == "AI_CORE":
        return "AI_CORE"
    if text == "AI_VECTOR_CORE":
        return "AI_VECTOR_CORE"
    if text == "MIX_AIC":
        return "MIX_AIC"
    if text == "MIX_AIV":
        return "MIX_AIV"
    if text == "HCCL":
        return "HCCL"
    if text == "DSA":
        return "DSA"
    raise ModelingError(f"accelerator_core_invalid:{value!r}")


def _req_bound_type(value: Any) -> BoundType:
    text = _req_nonempty_str(value, "bound_type_required")
    if text == "Compute Bound (Cube)":
        return "Compute Bound (Cube)"
    if text == "Compute Bound (Vector)":
        return "Compute Bound (Vector)"
    if text == "Memory Bound (HBM)":
        return "Memory Bound (HBM)"
    if text == "Communication Bound":
        return "Communication Bound"
    if text == "Latency Bound":
        return "Latency Bound"
    raise ModelingError(f"bound_type_invalid:{value!r}")


def _req_case_status(value: Any) -> CampaignCaseStatus:
    text = _req_nonempty_str(value, "case_status_required").lower()
    if text == "ok":
        return "ok"
    if text == "abi_rejected":
        return "abi_rejected"
    if text == "abi_unsupported":
        return "abi_unsupported"
    if text == "over_budget":
        return "over_budget"
    if text == "failed":
        return "failed"
    if text == "unsupported":
        return "unsupported"
    if text == "blocked":
        return "blocked"
    raise ModelingError(f"case_status_invalid:{value!r}")


def _freeze_str_map(value: Mapping[str, Any] | None, code: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ModelingError(code)
    out: dict[str, str] = {}
    for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
        out[_req_nonempty_str(str(k), code)] = _req_nonempty_str(str(v), f"{code}:{k}")
    return MappingProxyType(out)


def _freeze_strides_map(
    value: Mapping[str, Sequence[int]] | None, code: str,
) -> Mapping[str, tuple[int, ...]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ModelingError(code)
    out: dict[str, tuple[int, ...]] = {}
    for k, seq in sorted(value.items(), key=lambda kv: str(kv[0])):
        if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
            raise ModelingError(f"{code}:{k}")
        out[_req_nonempty_str(str(k), code)] = tuple(_req_int(v, f"{code}:{k}", minimum=0) for v in seq)
    return MappingProxyType(out)


def _freeze_int_map(
    value: Mapping[str, int] | None, code: str, *, minimum: int = 0,
) -> Mapping[str, int]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ModelingError(code)
    out: dict[str, int] = {}
    for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
        out[_req_nonempty_str(str(k), code)] = _req_int(v, f"{code}:{k}", minimum=minimum)
    return MappingProxyType(out)


def _freeze_opt_int_tuple(
    value: Sequence[int] | None, code: str, *, minimum: int = 0,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelingError(code)
    return tuple(_req_int(v, code, minimum=minimum) for v in value)


def _freeze_opt_str_tuple(value: Sequence[str] | None, code: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelingError(code)
    return tuple(_req_nonempty_str(v, code) for v in value)


def _freeze_json_mapping(value: Mapping[str, Any] | None, code: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ModelingError(code)
    frozen = _freeze_json(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_opt_json_mapping(value: Mapping[str, Any] | None, code: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _freeze_json_mapping(value, code)


def _freeze_row_tuples(value: Sequence[Sequence[Any]] | None, code: str) -> tuple[tuple[Any, ...], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelingError(code)
    rows: list[tuple[Any, ...]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ModelingError(code)
        rows.append(tuple(_freeze_json(item) for item in row))
    return tuple(rows)


def _freeze_opt_row_tuples(
    value: Sequence[Sequence[Any]] | None, code: str,
) -> tuple[tuple[Any, ...], ...] | None:
    if value is None:
        return None
    return _freeze_row_tuples(value, code)


def _assert_no_forbidden_key_fields_recursive(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            low = key.lower()
            if low in FORBIDDEN_CANONICAL_KEY_FIELDS or low.startswith("predicted_"):
                raise ModelingError(f"canonical_key_forbidden_field:{path}.{key}")
            _assert_no_forbidden_key_fields_recursive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_no_forbidden_key_fields_recursive(item, f"{path}[{idx}]")


def _freeze_opt_node_dicts(
    value: Sequence[Mapping[str, Any]] | None, code: str,
) -> tuple[Mapping[str, Any], ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelingError(code)
    out: list[Mapping[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ModelingError(code)
        item_keys = {str(k) for k in item}
        if _CHILD_KERNEL_KEY_MARKERS <= item_keys:
            canonical_child = CanonicalKernelKey.from_mapping(item).to_canonical_dict()
            out.append(_freeze_json_mapping(canonical_child, code))
        elif "canonical_key" in item and isinstance(item["canonical_key"], Mapping):
            _assert_no_forbidden_key_fields_recursive(item, f"$.timed_nodes[{idx}]")
            normalized = dict(item)
            normalized["canonical_key"] = CanonicalKernelKey.from_mapping(item["canonical_key"]).to_canonical_dict()
            out.append(_freeze_json_mapping(normalized, code))
        else:
            _assert_no_forbidden_key_fields_recursive(item, f"$.timed_nodes[{idx}]")
            out.append(_freeze_json_mapping(item, code))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CanonicalKernelKey:
    """Fixed 31-field model-agnostic physical execution signature (v2)."""

    kind: KernelKeyKind
    sku: str
    soc_arch: SocArch
    abi_backend: str
    upstream_abi: str
    bench_builder: str
    canonical_shapes: Mapping[str, Any]
    dtypes: Mapping[str, str]
    layouts: Mapping[str, str]
    capture_mode: CaptureMode
    residency: ResidencyMode = "cold_hbm"
    sig_version: int = CANONICAL_KEY_SIG_VERSION
    strides: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: MappingProxyType({}))
    storage_offsets: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    kv_allocator_block_size: int | None = None
    kv_kernel_block_size: int | None = None
    cp_kv_cache_interleave_size: int | None = None
    page_geometry: Mapping[str, Any] | None = None
    per_expert_token_counts: tuple[int, ...] | None = None
    per_rank_token_counts: tuple[int, ...] | None = None
    aclgraph_padded_bucket: int | None = None
    workspace_policy: str = "default"
    stream_dependencies: tuple[tuple[Any, ...], ...] = ()
    quant_meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    op_attrs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    ordered_rank_placement: tuple[int, ...] | None = None
    link_tiers: tuple[str, ...] | None = None
    collective_algorithm: str | None = None
    split_sizes: tuple[int, ...] | None = None
    timed_nodes: tuple[Mapping[str, Any], ...] | None = None
    edges: tuple[tuple[Any, ...], ...] | None = None

    def __post_init__(self) -> None:
        sig_ver = _req_int(self.sig_version, "canonical_key_sig_version_invalid")
        if sig_ver != CANONICAL_KEY_SIG_VERSION:
            raise ModelingError(f"canonical_key_sig_version_must_be_2:{sig_ver}")
        object.__setattr__(self, "sig_version", sig_ver)
        object.__setattr__(self, "kind", _req_key_kind(self.kind))
        object.__setattr__(self, "sku", _req_nonempty_str(self.sku, "canonical_key_sku_required"))
        object.__setattr__(self, "soc_arch", _req_soc_arch(self.soc_arch, "canonical_key_soc_arch_invalid"))
        abi_backend = _req_nonempty_str(self.abi_backend, "canonical_key_abi_backend_required").lower()
        if abi_backend not in HARDWARE_USABLE_ABI_BACKENDS:
            raise ModelingError(f"canonical_key_abi_backend_invalid:{abi_backend}")
        object.__setattr__(self, "abi_backend", abi_backend)
        object.__setattr__(self, "upstream_abi", _req_nonempty_str(self.upstream_abi, "canonical_key_upstream_abi_required"))
        object.__setattr__(self, "bench_builder", _req_nonempty_str(self.bench_builder, "canonical_key_builder_required"))
        object.__setattr__(self, "canonical_shapes", _freeze_json_mapping(self.canonical_shapes, "canonical_key_shapes_invalid"))
        object.__setattr__(self, "strides", _freeze_strides_map(self.strides, "canonical_key_strides_invalid"))
        object.__setattr__(self, "storage_offsets", _freeze_int_map(self.storage_offsets, "canonical_key_offsets_invalid", minimum=0))
        object.__setattr__(self, "dtypes", _freeze_str_map(self.dtypes, "canonical_key_dtypes_invalid"))
        object.__setattr__(self, "layouts", _freeze_str_map(self.layouts, "canonical_key_layouts_invalid"))
        object.__setattr__(self, "kv_allocator_block_size", _opt_int(self.kv_allocator_block_size, "canonical_key_kv_alloc_blk_invalid", minimum=1))
        object.__setattr__(self, "kv_kernel_block_size", _opt_int(self.kv_kernel_block_size, "canonical_key_kv_kern_blk_invalid", minimum=1))
        object.__setattr__(self, "cp_kv_cache_interleave_size", _opt_int(self.cp_kv_cache_interleave_size, "canonical_key_cp_interleave_invalid", minimum=1))
        object.__setattr__(self, "page_geometry", _freeze_opt_json_mapping(self.page_geometry, "canonical_key_page_geometry_invalid"))
        object.__setattr__(self, "per_expert_token_counts", _freeze_opt_int_tuple(self.per_expert_token_counts, "canonical_key_expert_counts_invalid", minimum=0))
        object.__setattr__(self, "per_rank_token_counts", _freeze_opt_int_tuple(self.per_rank_token_counts, "canonical_key_rank_counts_invalid", minimum=0))
        object.__setattr__(self, "capture_mode", _req_capture_mode(self.capture_mode))
        object.__setattr__(self, "aclgraph_padded_bucket", _opt_int(self.aclgraph_padded_bucket, "canonical_key_aclgraph_bucket_invalid", minimum=1))
        object.__setattr__(self, "workspace_policy", _req_nonempty_str(self.workspace_policy, "canonical_key_workspace_policy_required"))
        object.__setattr__(self, "stream_dependencies", _freeze_row_tuples(self.stream_dependencies, "canonical_key_stream_deps_invalid"))
        object.__setattr__(self, "residency", _req_residency(self.residency))
        object.__setattr__(self, "quant_meta", _freeze_json_mapping(self.quant_meta, "canonical_key_quant_meta_invalid"))
        object.__setattr__(self, "op_attrs", _freeze_json_mapping(self.op_attrs, "canonical_key_op_attrs_invalid"))
        object.__setattr__(self, "ordered_rank_placement", _freeze_opt_int_tuple(self.ordered_rank_placement, "canonical_key_rank_placement_invalid", minimum=0))
        object.__setattr__(self, "link_tiers", _freeze_opt_str_tuple(self.link_tiers, "canonical_key_link_tiers_invalid"))
        if self.collective_algorithm is not None:
            object.__setattr__(self, "collective_algorithm", _req_nonempty_str(self.collective_algorithm, "canonical_key_collective_algo_invalid"))
        object.__setattr__(self, "split_sizes", _freeze_opt_int_tuple(self.split_sizes, "canonical_key_split_sizes_invalid", minimum=0))
        object.__setattr__(self, "timed_nodes", _freeze_opt_node_dicts(self.timed_nodes, "canonical_key_timed_nodes_invalid"))
        object.__setattr__(self, "edges", _freeze_opt_row_tuples(self.edges, "canonical_key_edges_invalid"))
        _assert_no_forbidden_key_fields_recursive(self.canonical_shapes, "$.canonical_shapes")
        if self.page_geometry is not None:
            _assert_no_forbidden_key_fields_recursive(self.page_geometry, "$.page_geometry")
        _assert_no_forbidden_key_fields_recursive(self.quant_meta, "$.quant_meta")
        _assert_no_forbidden_key_fields_recursive(self.op_attrs, "$.op_attrs")
        _assert_no_forbidden_key_fields_recursive(self.stream_dependencies, "$.stream_dependencies")
        if self.edges is not None:
            _assert_no_forbidden_key_fields_recursive(self.edges, "$.edges")

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the exact fixed 31-field canonical JSON payload."""
        return {
            "sig_version": self.sig_version,
            "kind": self.kind,
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "abi_backend": self.abi_backend,
            "upstream_abi": self.upstream_abi,
            "bench_builder": self.bench_builder,
            "canonical_shapes": _plain_json(self.canonical_shapes),
            "strides": {k: list(v) for k, v in sorted(self.strides.items())},
            "storage_offsets": {k: int(v) for k, v in sorted(self.storage_offsets.items())},
            "dtypes": dict(sorted(self.dtypes.items())),
            "layouts": dict(sorted(self.layouts.items())),
            "kv_allocator_block_size": self.kv_allocator_block_size,
            "kv_kernel_block_size": self.kv_kernel_block_size,
            "cp_kv_cache_interleave_size": self.cp_kv_cache_interleave_size,
            "page_geometry": _plain_json(self.page_geometry) if self.page_geometry is not None else None,
            "per_expert_token_counts": list(self.per_expert_token_counts) if self.per_expert_token_counts is not None else None,
            "per_rank_token_counts": list(self.per_rank_token_counts) if self.per_rank_token_counts is not None else None,
            "capture_mode": self.capture_mode,
            "aclgraph_padded_bucket": self.aclgraph_padded_bucket,
            "workspace_policy": self.workspace_policy,
            "stream_dependencies": [_plain_json(row) for row in self.stream_dependencies],
            "residency": self.residency,
            "quant_meta": _plain_json(self.quant_meta),
            "op_attrs": _plain_json(self.op_attrs),
            "ordered_rank_placement": list(self.ordered_rank_placement) if self.ordered_rank_placement is not None else None,
            "link_tiers": list(self.link_tiers) if self.link_tiers is not None else None,
            "collective_algorithm": self.collective_algorithm,
            "split_sizes": list(self.split_sizes) if self.split_sizes is not None else None,
            "timed_nodes": [_plain_json(n) for n in self.timed_nodes] if self.timed_nodes is not None else None,
            "edges": [_plain_json(e) for e in self.edges] if self.edges is not None else None,
        }

    def digest(self) -> str:
        """Return 16-hex SHA-256 prefix over the exact 31-field canonical JSON."""
        return canonical_sha256(self.to_canonical_dict())[:CANONICAL_DIGEST_HEX_LEN]

    def case_id(self) -> str:
        """Return canonical ``k2-<16hex>`` case identifier."""
        return f"{CASE_ID_PREFIX}{self.digest()}"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CanonicalKernelKey:
        """Parse and validate a 31-field canonical key, rejecting forbidden keys."""
        if not isinstance(raw, Mapping):
            raise ModelingError("canonical_key_requires_mapping")
        allowed = frozenset(CANONICAL_KERNEL_KEY_FIELDS)
        for key_name in raw:
            key_str = str(key_name)
            low = key_str.lower()
            if low in FORBIDDEN_CANONICAL_KEY_FIELDS or low.startswith("predicted_"):
                raise ModelingError(f"canonical_key_forbidden_field:{key_str}")
            if key_str not in allowed:
                raise ModelingError(f"canonical_key_unknown_field:{key_str}")
        return cls(
            sig_version=raw.get("sig_version", CANONICAL_KEY_SIG_VERSION),
            kind=raw["kind"],
            sku=raw["sku"],
            soc_arch=raw["soc_arch"],
            abi_backend=raw["abi_backend"],
            upstream_abi=raw["upstream_abi"],
            bench_builder=raw["bench_builder"],
            canonical_shapes=raw.get("canonical_shapes", {}),
            strides=raw.get("strides", {}),
            storage_offsets=raw.get("storage_offsets", {}),
            dtypes=raw.get("dtypes", {}),
            layouts=raw.get("layouts", {}),
            kv_allocator_block_size=raw.get("kv_allocator_block_size"),
            kv_kernel_block_size=raw.get("kv_kernel_block_size"),
            cp_kv_cache_interleave_size=raw.get("cp_kv_cache_interleave_size"),
            page_geometry=raw.get("page_geometry"),
            per_expert_token_counts=raw.get("per_expert_token_counts"),
            per_rank_token_counts=raw.get("per_rank_token_counts"),
            capture_mode=raw["capture_mode"],
            aclgraph_padded_bucket=raw.get("aclgraph_padded_bucket"),
            workspace_policy=raw.get("workspace_policy", "default"),
            stream_dependencies=raw.get("stream_dependencies", ()),
            residency=raw.get("residency", "cold_hbm"),
            quant_meta=raw.get("quant_meta", {}),
            op_attrs=raw.get("op_attrs", {}),
            ordered_rank_placement=raw.get("ordered_rank_placement"),
            link_tiers=raw.get("link_tiers"),
            collective_algorithm=raw.get("collective_algorithm"),
            split_sizes=raw.get("split_sizes"),
            timed_nodes=raw.get("timed_nodes"),
            edges=raw.get("edges"),
        )


def canonical_key_digest(key: CanonicalKernelKey | Mapping[str, Any]) -> str:
    """Compute exact 16-hex canonical execution digest for a ``CanonicalKernelKey``."""
    parsed = key if isinstance(key, CanonicalKernelKey) else CanonicalKernelKey.from_mapping(key)
    return parsed.digest()


def canonical_case_id(key: CanonicalKernelKey | Mapping[str, Any]) -> str:
    """Compute exact ``k2-<16hex>`` case identifier for a ``CanonicalKernelKey``."""
    return f"{CASE_ID_PREFIX}{canonical_key_digest(key)}"


@dataclass(frozen=True, slots=True)
class ReverseIndexEntry:
    """Lossless occurrence-level reverse index binding a case back to matrix cells."""

    cell_id: str
    stage: str
    scope: ModelScopeKind
    native_layer: int
    replica: int
    rank: int | None
    op_id: str
    model_id: str = ""
    sku: str = ""
    phase: PhaseKind = "decode"
    seq_len: int = 0
    concurrency: int = 1
    parallel_arm: str = ""
    layer_archetype_id: str = ""
    occurrence_count: int = 1
    model_step_share_pct: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _req_nonempty_str(self.cell_id, "reverse_index_cell_id_required"))
        object.__setattr__(self, "stage", _req_nonempty_str(self.stage, "reverse_index_stage_required"))
        object.__setattr__(self, "scope", _req_model_scope(self.scope))
        object.__setattr__(self, "native_layer", _req_int(self.native_layer, "reverse_index_native_layer_invalid", minimum=0))
        object.__setattr__(self, "replica", _req_int(self.replica, "reverse_index_replica_invalid", minimum=0))
        object.__setattr__(self, "rank", _opt_int(self.rank, "reverse_index_rank_invalid", minimum=0))
        object.__setattr__(self, "op_id", _req_nonempty_str(self.op_id, "reverse_index_op_id_required"))
        object.__setattr__(self, "phase", _req_phase(self.phase))
        object.__setattr__(self, "seq_len", _req_int(self.seq_len, "reverse_index_seq_len_invalid", minimum=0))
        object.__setattr__(self, "concurrency", _req_int(self.concurrency, "reverse_index_concurrency_invalid", minimum=1))
        object.__setattr__(self, "occurrence_count", _req_int(self.occurrence_count, "reverse_index_occ_count_invalid", minimum=1))
        object.__setattr__(self, "model_step_share_pct", _req_float(self.model_step_share_pct, "reverse_index_share_invalid", minimum=0.0))

    def occurrence_key(self) -> tuple[str, str, str, int, int, int | None, str]:
        """Exact ``(cell_id, stage, scope, native_layer, replica, rank, op_id)`` key."""
        return (
            self.cell_id,
            self.stage,
            self.scope,
            self.native_layer,
            self.replica,
            self.rank,
            self.op_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "stage": self.stage,
            "scope": self.scope,
            "native_layer": self.native_layer,
            "replica": self.replica,
            "rank": self.rank,
            "op_id": self.op_id,
            "model_id": self.model_id,
            "sku": self.sku,
            "phase": self.phase,
            "seq_len": self.seq_len,
            "concurrency": self.concurrency,
            "parallel_arm": self.parallel_arm,
            "layer_archetype_id": self.layer_archetype_id,
            "occurrence_count": self.occurrence_count,
            "model_step_share_pct": self.model_step_share_pct,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ReverseIndexEntry:
        if not isinstance(raw, Mapping):
            raise ModelingError("reverse_index_entry_requires_mapping")
        return cls(
            cell_id=raw["cell_id"],
            stage=raw["stage"],
            scope=raw.get("scope", "target"),
            native_layer=raw["native_layer"],
            replica=raw.get("replica", 0),
            rank=raw.get("rank"),
            op_id=raw["op_id"],
            model_id=str(raw.get("model_id") or ""),
            sku=str(raw.get("sku") or ""),
            phase=raw.get("phase", "decode"),
            seq_len=raw.get("seq_len", 0),
            concurrency=raw.get("concurrency", 1),
            parallel_arm=str(raw.get("parallel_arm") or ""),
            layer_archetype_id=str(raw.get("layer_archetype_id") or ""),
            occurrence_count=raw.get("occurrence_count", 1),
            model_step_share_pct=raw.get("model_step_share_pct", 0.0),
        )


@dataclass(frozen=True, slots=True)
class BenchPlanCaseV2:
    """Single capture-mode benchmark case in ``mindie-bench-plan/2``."""

    acquisition_uid: str
    case_id: str
    canonical_key_digest: str
    canonical_key: CanonicalKernelKey
    kind: KernelKeyKind
    level: str
    boundary: str
    split: PlanSplit
    phase: str
    stage: str
    capture_mode: CaptureMode
    residency: ResidencyMode = "cold_hbm"
    clock_operator: ClockOperator = "elapsed_span"
    work_flops: int = 0
    work_bytes: int = 0
    nodes: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[tuple[Any, ...], ...] = ()
    timed_boundary: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    identities: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "acquisition_uid", _req_nonempty_str(self.acquisition_uid, "plan_case_acq_uid_required"))
        if not isinstance(self.canonical_key, CanonicalKernelKey):
            raise ModelingError("plan_case_canonical_key_invalid")
        expected_digest = self.canonical_key.digest()
        digest = _validate_digest16(self.canonical_key_digest, "plan_case_digest_invalid")
        if digest != expected_digest:
            raise ModelingError(f"plan_case_digest_mismatch:{digest}!={expected_digest}")
        expected_cid = f"{CASE_ID_PREFIX}{expected_digest}"
        cid = _req_nonempty_str(self.case_id, "plan_case_id_required")
        if cid != expected_cid:
            raise ModelingError(f"plan_case_id_mismatch:{cid}!={expected_cid}")
        object.__setattr__(self, "case_id", cid)
        object.__setattr__(self, "canonical_key_digest", digest)
        object.__setattr__(self, "kind", _req_key_kind(self.kind))
        if self.kind != self.canonical_key.kind:
            raise ModelingError(f"plan_case_kind_mismatch:{self.kind}!={self.canonical_key.kind}")
        object.__setattr__(self, "level", _req_nonempty_str(self.level, "plan_case_level_required"))
        object.__setattr__(self, "boundary", _req_nonempty_str(self.boundary, "plan_case_boundary_required"))
        object.__setattr__(self, "split", _req_plan_split(self.split))
        object.__setattr__(self, "phase", _req_nonempty_str(self.phase, "plan_case_phase_required"))
        object.__setattr__(self, "stage", _req_nonempty_str(self.stage, "plan_case_stage_required"))
        cm = _req_capture_mode(self.capture_mode)
        if cm != self.canonical_key.capture_mode:
            raise ModelingError(f"plan_case_capture_mode_mismatch:{cm}!={self.canonical_key.capture_mode}")
        object.__setattr__(self, "capture_mode", cm)
        res = _req_residency(self.residency)
        if res != self.canonical_key.residency:
            raise ModelingError(f"plan_case_residency_mismatch:{res}!={self.canonical_key.residency}")
        object.__setattr__(self, "residency", res)
        object.__setattr__(self, "clock_operator", _req_clock_operator(self.clock_operator))
        object.__setattr__(self, "work_flops", _req_int(self.work_flops, "plan_case_work_flops_invalid", minimum=0))
        object.__setattr__(self, "work_bytes", _req_int(self.work_bytes, "plan_case_work_bytes_invalid", minimum=0))
        object.__setattr__(self, "nodes", tuple(_freeze_json_mapping(n, "plan_case_nodes_invalid") for n in self.nodes))
        object.__setattr__(self, "edges", _freeze_row_tuples(self.edges, "plan_case_edges_invalid"))
        object.__setattr__(self, "timed_boundary", _freeze_json_mapping(self.timed_boundary, "plan_case_timed_boundary_invalid"))
        object.__setattr__(self, "identities", _freeze_json_mapping(self.identities, "plan_case_identities_invalid"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "acquisition_uid": self.acquisition_uid,
            "case_id": self.case_id,
            "canonical_key_digest": self.canonical_key_digest,
            "canonical_key": self.canonical_key.to_canonical_dict(),
            "kind": self.kind,
            "level": self.level,
            "boundary": self.boundary,
            "split": self.split,
            "phase": self.phase,
            "stage": self.stage,
            "capture_mode": self.capture_mode,
            "residency": self.residency,
            "clock_operator": self.clock_operator,
            "work_flops": self.work_flops,
            "work_bytes": self.work_bytes,
            "nodes": [_plain_json(n) for n in self.nodes],
            "edges": [_plain_json(e) for e in self.edges],
            "timed_boundary": _plain_json(self.timed_boundary),
            "identities": _plain_json(self.identities),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchPlanCaseV2:
        if not isinstance(raw, Mapping):
            raise ModelingError("plan_case_requires_mapping")
        for key in raw:
            low = str(key).lower()
            if low.startswith("predicted_"):
                raise ModelingError(f"plan_case_forbidden_prediction_key:{key}")
        ck = CanonicalKernelKey.from_mapping(raw["canonical_key"])
        return cls(
            acquisition_uid=raw["acquisition_uid"],
            case_id=raw["case_id"],
            canonical_key_digest=raw["canonical_key_digest"],
            canonical_key=ck,
            kind=raw["kind"],
            level=raw.get("level", "L1"),
            boundary=raw.get("boundary", "device_kernel"),
            split=raw["split"],
            phase=raw["phase"],
            stage=raw.get("stage", "s0_local"),
            capture_mode=raw.get("capture_mode", ck.capture_mode),
            residency=raw.get("residency", ck.residency),
            clock_operator=raw.get("clock_operator", "elapsed_span"),
            work_flops=raw.get("work_flops", 0),
            work_bytes=raw.get("work_bytes", 0),
            nodes=tuple(raw.get("nodes", ())),
            edges=tuple(raw.get("edges", ())),
            timed_boundary=raw.get("timed_boundary", {}),
            identities=raw.get("identities", {}),
        )


def compute_plan_sha256(plan_payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 seal over ``mindie-bench-plan/2`` excluding ``plan_sha256``."""
    body = {str(k): v for k, v in plan_payload.items() if str(k) != "plan_sha256"}
    return canonical_sha256(body)


@dataclass(frozen=True, slots=True)
class BenchPlanV2:
    """Frozen ``mindie-bench-plan/2`` cross-repo benchmark plan."""

    plan_id: str
    sku: str
    soc_arch: SocArch
    cases: tuple[BenchPlanCaseV2, ...]
    reverse_index: Mapping[str, tuple[ReverseIndexEntry, ...]]
    plan_sha256: str = ""
    schema: Literal["mindie-bench-plan/2"] = BENCH_PLAN_V2_SCHEMA
    plan_kind: str = "model_matrix"
    stage: str = "s0_local"
    cache_partition: CachePartition = "shared"
    world_size: int = 1
    training_manifest_sha256: str | None = None
    extra_fields: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.schema != BENCH_PLAN_V2_SCHEMA:
            raise ModelingError(f"bench_plan_schema_invalid:{self.schema!r}")
        object.__setattr__(self, "plan_id", _req_nonempty_str(self.plan_id, "bench_plan_id_required"))
        object.__setattr__(self, "sku", _req_nonempty_str(self.sku, "bench_plan_sku_required"))
        object.__setattr__(self, "soc_arch", _req_soc_arch(self.soc_arch, "bench_plan_soc_arch_invalid"))
        object.__setattr__(self, "plan_kind", _req_nonempty_str(self.plan_kind, "bench_plan_kind_required"))
        object.__setattr__(self, "stage", _req_nonempty_str(self.stage, "bench_plan_stage_required"))
        object.__setattr__(self, "cache_partition", _req_cache_partition(self.cache_partition))
        object.__setattr__(self, "world_size", _req_int(self.world_size, "bench_plan_world_size_invalid", minimum=1))
        cases_t = tuple(self.cases)
        seen_cids: set[str] = set()
        for case in cases_t:
            if not isinstance(case, BenchPlanCaseV2):
                raise ModelingError("bench_plan_case_type_invalid")
            if case.case_id in seen_cids:
                raise ModelingError(f"bench_plan_duplicate_case_id:{case.case_id}")
            seen_cids.add(case.case_id)
        object.__setattr__(self, "cases", cases_t)
        rev: dict[str, tuple[ReverseIndexEntry, ...]] = {}
        for cid, entries in sorted(self.reverse_index.items(), key=lambda kv: str(kv[0])):
            parsed_entries: list[ReverseIndexEntry] = []
            for entry in entries:
                if not isinstance(entry, ReverseIndexEntry):
                    raise ModelingError("bench_plan_reverse_index_entry_invalid")
                parsed_entries.append(entry)
            rev[str(cid)] = tuple(parsed_entries)
        object.__setattr__(self, "reverse_index", MappingProxyType(rev))
        if self.training_manifest_sha256 is not None:
            object.__setattr__(
                self, "training_manifest_sha256",
                _validate_sha256_hex(self.training_manifest_sha256, "bench_plan_training_manifest_sha256_invalid"),
            )
        if self.plan_sha256:
            _validate_sha256_hex(self.plan_sha256, "bench_plan_sha256_invalid")
        extra = _freeze_json_mapping(self.extra_fields, "bench_plan_extra_fields_invalid")
        assert_no_forbidden_prediction_keys(extra, "$.extra_fields")
        for key_name in extra:
            if key_name in _BENCH_PLAN_KNOWN_FIELDS:
                raise ModelingError(f"bench_plan_extra_field_collision:{key_name}")
        object.__setattr__(self, "extra_fields", extra)

    def to_dict(self, *, include_seal: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **_plain_json(self.extra_fields),
            "schema": self.schema,
            "plan_id": self.plan_id,
            "plan_kind": self.plan_kind,
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "stage": self.stage,
            "cache_partition": self.cache_partition,
            "world_size": self.world_size,
            "training_manifest_sha256": self.training_manifest_sha256,
            "cases": [c.to_dict() for c in self.cases],
            "reverse_index": {
                cid: [entry.to_dict() for entry in entries]
                for cid, entries in sorted(self.reverse_index.items())
            },
        }
        if include_seal:
            payload["plan_sha256"] = self.plan_sha256 or compute_plan_sha256(payload)
        return payload

    def with_computed_seal(self) -> BenchPlanV2:
        seal = compute_plan_sha256(self.to_dict(include_seal=False))
        return BenchPlanV2(
            plan_id=self.plan_id,
            sku=self.sku,
            soc_arch=self.soc_arch,
            cases=self.cases,
            reverse_index=self.reverse_index,
            plan_sha256=seal,
            schema=self.schema,
            plan_kind=self.plan_kind,
            stage=self.stage,
            cache_partition=self.cache_partition,
            world_size=self.world_size,
            training_manifest_sha256=self.training_manifest_sha256,
            extra_fields=self.extra_fields,
        )

    def verify_seal(self) -> None:
        expected = compute_plan_sha256(self.to_dict(include_seal=False))
        if not self.plan_sha256 or self.plan_sha256 != expected:
            raise ModelingError(f"bench_plan_sha256_mismatch:{self.plan_sha256}!={expected}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, verify_seal: bool = True) -> BenchPlanV2:
        if not isinstance(raw, Mapping):
            raise ModelingError("bench_plan_requires_mapping")
        assert_no_forbidden_prediction_keys(raw, "$")
        if raw.get("schema") != BENCH_PLAN_V2_SCHEMA:
            raise ModelingError(f"bench_plan_schema_invalid:{raw.get('schema')!r}")
        if verify_seal:
            recorded = _validate_sha256_hex(raw.get("plan_sha256"), "bench_plan_sha256_missing")
            expected_raw = compute_plan_sha256(raw)
            if recorded != expected_raw:
                raise ModelingError(f"bench_plan_sha256_mismatch:{recorded}!={expected_raw}")
        cases = tuple(BenchPlanCaseV2.from_mapping(c) for c in raw.get("cases", ()))
        raw_rev = raw.get("reverse_index", {})
        if not isinstance(raw_rev, Mapping):
            raise ModelingError("bench_plan_reverse_index_invalid")
        rev = {
            str(cid): tuple(ReverseIndexEntry.from_mapping(e) for e in entries)
            for cid, entries in raw_rev.items()
        }
        extra = {
            str(k): v for k, v in raw.items()
            if str(k) not in _BENCH_PLAN_KNOWN_FIELDS
        }
        plan = cls(
            plan_id=raw["plan_id"],
            sku=raw["sku"],
            soc_arch=raw["soc_arch"],
            cases=cases,
            reverse_index=rev,
            plan_sha256=str(raw.get("plan_sha256") or ""),
            schema=BENCH_PLAN_V2_SCHEMA,
            plan_kind=raw.get("plan_kind", "model_matrix"),
            stage=raw.get("stage", "s0_local"),
            cache_partition=raw.get("cache_partition", "shared"),
            world_size=raw.get("world_size", 1),
            training_manifest_sha256=raw.get("training_manifest_sha256"),
            extra_fields=extra,
        )
        if verify_seal:
            plan.verify_seal()
        return plan


def compute_env_fingerprint_id(static_bucket_key: Mapping[str, Any]) -> str:
    """Compute canonical ``env-<sku_slug>-cann<cann_short>-<sha256[:16]>`` ID."""
    sku = str(static_bucket_key.get("sku") or "")
    cann_ver = str(static_bucket_key.get("cann_version") or "")
    sku_slug = _SLUG_RE.sub("_", sku.lower()).strip("_") or "unknown"
    cann_short = _CANN_SLUG_RE.sub("_", cann_ver.lower()).strip("_") or "unknown"
    ordered = {k: static_bucket_key.get(k) for k in STATIC_ENV_BUCKET_FIELDS}
    digest = canonical_sha256(ordered)[:ENV_DIGEST_HEX_LEN]
    return f"env-{sku_slug}-cann{cann_short}-{digest}"


@dataclass(frozen=True, slots=True)
class BenchEnvFingerprint:
    """Frozen ``mindie-bench-env/1`` hardware + software-stack fingerprint."""

    sku: str
    soc_arch: SocArch
    hw_geometry: Mapping[str, Any]
    driver_version: str
    cann_version: str
    cann_innerversion: str
    cann_packages: Mapping[str, str]
    torch_version: str
    torch_npu_version: str
    va_commit: str
    triton_version: str
    custom_op_libs: Mapping[str, str]
    profiler_version: str
    timing_protocol_id: str
    cache_partition: CachePartition = "shared"
    observed_clock_mhz: float | None = None
    clock_source: str = "unavailable"
    env_fingerprint_id: str = ""
    schema: Literal["mindie-bench-env/1"] = BENCH_ENV_V1_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BENCH_ENV_V1_SCHEMA:
            raise ModelingError(f"bench_env_schema_invalid:{self.schema!r}")
        object.__setattr__(self, "sku", _req_nonempty_str(self.sku, "bench_env_sku_required"))
        object.__setattr__(self, "soc_arch", _req_soc_arch(self.soc_arch, "bench_env_soc_arch_invalid"))
        object.__setattr__(self, "hw_geometry", _freeze_json_mapping(self.hw_geometry, "bench_env_hw_geometry_invalid"))
        for fname in (
            "driver_version", "cann_version", "cann_innerversion",
            "torch_version", "torch_npu_version", "va_commit",
            "triton_version", "profiler_version", "timing_protocol_id", "clock_source",
        ):
            object.__setattr__(self, fname, _req_nonempty_str(getattr(self, fname), f"bench_env_{fname}_required"))
        object.__setattr__(self, "cann_packages", _freeze_str_map(self.cann_packages, "bench_env_cann_packages_invalid"))
        object.__setattr__(self, "custom_op_libs", _freeze_str_map(self.custom_op_libs, "bench_env_custom_op_libs_invalid"))
        object.__setattr__(self, "cache_partition", _req_cache_partition(self.cache_partition))
        object.__setattr__(self, "observed_clock_mhz", _opt_float(self.observed_clock_mhz, "bench_env_clock_invalid", strictly_positive=True))
        expected_id = compute_env_fingerprint_id(self.static_bucket_key())
        if self.env_fingerprint_id:
            if self.env_fingerprint_id != expected_id:
                raise ModelingError(f"bench_env_id_mismatch:{self.env_fingerprint_id}!={expected_id}")
        else:
            object.__setattr__(self, "env_fingerprint_id", expected_id)

    def static_bucket_key(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "hw_geometry": _plain_json(self.hw_geometry),
            "driver_version": self.driver_version,
            "cann_version": self.cann_version,
            "cann_innerversion": self.cann_innerversion,
            "cann_packages": dict(sorted(self.cann_packages.items())),
            "torch_version": self.torch_version,
            "torch_npu_version": self.torch_npu_version,
            "va_commit": self.va_commit,
            "triton_version": self.triton_version,
            "custom_op_libs": dict(sorted(self.custom_op_libs.items())),
            "profiler_version": self.profiler_version,
            "timing_protocol_id": self.timing_protocol_id,
            "cache_partition": self.cache_partition,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "env_fingerprint_id": self.env_fingerprint_id,
            **self.static_bucket_key(),
            "observed_clock_mhz": self.observed_clock_mhz,
            "clock_source": self.clock_source,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchEnvFingerprint:
        if not isinstance(raw, Mapping):
            raise ModelingError("bench_env_requires_mapping")
        assert_no_forbidden_prediction_keys(raw, "$.version_fingerprint")
        return cls(
            sku=raw["sku"],
            soc_arch=raw["soc_arch"],
            hw_geometry=raw.get("hw_geometry", {}),
            driver_version=raw["driver_version"],
            cann_version=raw["cann_version"],
            cann_innerversion=raw["cann_innerversion"],
            cann_packages=raw.get("cann_packages", {}),
            torch_version=raw["torch_version"],
            torch_npu_version=raw["torch_npu_version"],
            va_commit=raw["va_commit"],
            triton_version=raw["triton_version"],
            custom_op_libs=raw.get("custom_op_libs", {}),
            profiler_version=raw["profiler_version"],
            timing_protocol_id=raw["timing_protocol_id"],
            cache_partition=raw.get("cache_partition", "shared"),
            observed_clock_mhz=raw.get("observed_clock_mhz"),
            clock_source=raw.get("clock_source", "unavailable"),
            env_fingerprint_id=str(raw.get("env_fingerprint_id") or ""),
            schema=BENCH_ENV_V1_SCHEMA,
        )


@dataclass(frozen=True, slots=True)
class BenchObservationV2:
    """Frozen ``mindie-bench-observation/2`` JSONL record (superset of v1)."""

    sample_id: str
    acquisition_uid: str
    case_id: str
    canonical_key_digest: str
    env_fingerprint_id: str
    run_id: str
    split: PlanSplit
    boundary: str
    status: CampaignCaseStatus
    qualified: bool
    timing_mode: str
    capture_mode: CaptureMode
    clock_operator: ClockOperator
    residency: ResidencyMode
    rank: int
    source: str
    duration_ms: float | None = None
    observed_clock_mhz: float | None = None
    hbm_peak_bytes: int | None = None
    cache_status: str = "measured"
    actual_executed_shape: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    identity: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    key: Mapping[str, Any] | None = None
    work: float | None = None
    work_bytes: int | None = None
    abi_rejected: bool = False
    abi_error: str | None = None
    unmodeled_hint: Mapping[str, Any] | None = None
    cold_coverage_verified: bool = True
    record_type: str = "observation"
    schema: Literal["mindie-bench-observation/2"] = BENCH_OBSERVATION_V2_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BENCH_OBSERVATION_V2_SCHEMA:
            raise ModelingError(f"bench_obs_schema_invalid:{self.schema!r}")
        object.__setattr__(self, "sample_id", _req_nonempty_str(self.sample_id, "bench_obs_sample_id_required"))
        object.__setattr__(self, "acquisition_uid", _req_nonempty_str(self.acquisition_uid, "bench_obs_acq_uid_required"))
        digest = _validate_digest16(self.canonical_key_digest, "bench_obs_digest_invalid")
        cid = _req_nonempty_str(self.case_id, "bench_obs_case_id_required")
        if cid != f"{CASE_ID_PREFIX}{digest}":
            raise ModelingError(f"bench_obs_case_id_mismatch:{cid}")
        object.__setattr__(self, "case_id", cid)
        object.__setattr__(self, "canonical_key_digest", digest)
        object.__setattr__(self, "env_fingerprint_id", _req_nonempty_str(self.env_fingerprint_id, "bench_obs_env_id_required"))
        object.__setattr__(self, "run_id", _req_nonempty_str(self.run_id, "bench_obs_run_id_required"))
        object.__setattr__(self, "split", _req_plan_split(self.split))
        object.__setattr__(self, "boundary", _req_nonempty_str(self.boundary, "bench_obs_boundary_required"))
        object.__setattr__(self, "status", _req_case_status(self.status))
        if not isinstance(self.qualified, bool):
            raise ModelingError("bench_obs_qualified_bool_required")
        object.__setattr__(self, "timing_mode", _req_nonempty_str(self.timing_mode, "bench_obs_timing_mode_required"))
        object.__setattr__(self, "capture_mode", _req_capture_mode(self.capture_mode))
        object.__setattr__(self, "clock_operator", _req_clock_operator(self.clock_operator))
        object.__setattr__(self, "residency", _req_residency(self.residency))
        object.__setattr__(self, "rank", _req_int(self.rank, "bench_obs_rank_int_required", minimum=0))
        object.__setattr__(self, "source", _req_nonempty_str(self.source, "bench_obs_source_required"))
        object.__setattr__(self, "duration_ms", _opt_float(self.duration_ms, "bench_obs_duration_invalid", strictly_positive=True))
        object.__setattr__(self, "observed_clock_mhz", _opt_float(self.observed_clock_mhz, "bench_obs_clock_invalid", strictly_positive=True))
        object.__setattr__(self, "hbm_peak_bytes", _opt_int(self.hbm_peak_bytes, "bench_obs_hbm_peak_invalid", minimum=0))
        object.__setattr__(self, "actual_executed_shape", _freeze_json_mapping(self.actual_executed_shape, "bench_obs_shape_invalid"))
        object.__setattr__(self, "identity", _freeze_json_mapping(self.identity, "bench_obs_identity_invalid"))
        object.__setattr__(self, "key", _freeze_opt_json_mapping(self.key, "bench_obs_key_invalid"))
        object.__setattr__(self, "work", _opt_float(self.work, "bench_obs_work_invalid", minimum=0.0))
        object.__setattr__(self, "work_bytes", _opt_int(self.work_bytes, "bench_obs_work_bytes_invalid", minimum=0))
        object.__setattr__(self, "unmodeled_hint", _freeze_opt_json_mapping(self.unmodeled_hint, "bench_obs_hint_invalid"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "sample_id": self.sample_id,
            "acquisition_uid": self.acquisition_uid,
            "case_id": self.case_id,
            "canonical_key_digest": self.canonical_key_digest,
            "env_fingerprint_id": self.env_fingerprint_id,
            "run_id": self.run_id,
            "split": self.split,
            "boundary": self.boundary,
            "status": self.status,
            "qualified": self.qualified,
            "timing_mode": self.timing_mode,
            "capture_mode": self.capture_mode,
            "clock_operator": self.clock_operator,
            "residency": self.residency,
            "rank": self.rank,
            "source": self.source,
            "duration_ms": self.duration_ms,
            "observed_clock_mhz": self.observed_clock_mhz,
            "hbm_peak_bytes": self.hbm_peak_bytes,
            "cache_status": self.cache_status,
            "actual_executed_shape": _plain_json(self.actual_executed_shape),
            "identity": _plain_json(self.identity),
            "key": _plain_json(self.key) if self.key is not None else None,
            "work": self.work,
            "work_bytes": self.work_bytes,
            "abi_rejected": self.abi_rejected,
            "abi_error": self.abi_error,
            "unmodeled_hint": _plain_json(self.unmodeled_hint) if self.unmodeled_hint is not None else None,
            "cold_coverage_verified": self.cold_coverage_verified,
        }


@dataclass(frozen=True, slots=True)
class RunShardDescriptor:
    """Immutable per-run observation shard entry in ``mindie-bench-campaign/1``."""

    run_id: str
    path: str
    observations_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _req_nonempty_str(self.run_id, "run_shard_run_id_required"))
        object.__setattr__(self, "path", _req_nonempty_str(self.path, "run_shard_path_required"))
        object.__setattr__(self, "observations_sha256", _validate_sha256_hex(self.observations_sha256, "run_shard_sha256_invalid"))
        object.__setattr__(self, "record_count", _req_int(self.record_count, "run_shard_record_count_invalid", minimum=0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "observations_sha256": self.observations_sha256,
            "record_count": self.record_count,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RunShardDescriptor:
        if not isinstance(raw, Mapping):
            raise ModelingError("run_shard_requires_mapping")
        return cls(
            run_id=raw["run_id"],
            path=raw["path"],
            observations_sha256=raw["observations_sha256"],
            record_count=raw["record_count"],
        )


CASE_RAW_SHA256_EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "raw_sha256",
    "cache_status",
    "split",
})


def compute_case_raw_sha256(case_payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over immutable silicon measurement fields.

    Excludes ``raw_sha256``, ``cache_status`` (flips ``measured -> hit`` on reuse),
    and plan-assigned ``split`` (sealed separately by ``campaign_sha256`` and
    validated by ``training_manifest_sha256``) so cache hits preserve original
    ``raw_sha256`` bit-for-bit across plans.
    """
    body = {str(k): v for k, v in case_payload.items() if str(k) not in CASE_RAW_SHA256_EXCLUDED_FIELDS}
    return canonical_sha256(body)


def compute_campaign_sha256(campaign_payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over ``mindie-bench-campaign/1`` excluding ``campaign_sha256``."""
    body = {str(k): v for k, v in campaign_payload.items() if str(k) != "campaign_sha256"}
    return canonical_sha256(body)


@dataclass(frozen=True, slots=True)
class CampaignCaseRecord:
    """Single reconciled silicon case record inside ``mindie-bench-campaign/1``."""

    acquisition_uid: str
    case_id: str
    canonical_key_digest: str
    boundary: str
    split: PlanSplit
    status: CampaignCaseStatus
    qualified: bool
    timing_mode: str
    capture_mode: CaptureMode
    clock_operator: ClockOperator
    residency: ResidencyMode
    rank: int
    participating_ranks: tuple[int, ...]
    timed_nodes: tuple[str, ...]
    p50_us: float
    p99_us: float
    samples_count: int
    observed_clock_mhz: float
    capture_uri: str
    raw_sha256: str
    p90_us: float | None = None
    min_us: float | None = None
    iqr_us: float | None = None
    cv_pct: float | None = None
    hbm_peak_bytes: int | None = None
    actual_executed_shape: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    cache_status: str = "measured"
    measured_in_run_id: str = ""
    run_shard: str = ""
    cold_coverage_verified: bool = True
    traced_interval_verified: bool = False
    family: str = ""
    geometry_key: str = ""
    upstream_abi: str = ""
    abi_backend: str = ""
    bench_builder: str = ""
    extra_fields: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "acquisition_uid", _req_nonempty_str(self.acquisition_uid, "campaign_case_acq_uid_required"))
        digest = _validate_digest16(self.canonical_key_digest, "campaign_case_digest_invalid")
        cid = _req_nonempty_str(self.case_id, "campaign_case_id_required")
        if cid != f"{CASE_ID_PREFIX}{digest}":
            raise ModelingError(f"campaign_case_id_mismatch:{cid}")
        object.__setattr__(self, "case_id", cid)
        object.__setattr__(self, "canonical_key_digest", digest)
        object.__setattr__(self, "boundary", _req_nonempty_str(self.boundary, "campaign_case_boundary_required"))
        object.__setattr__(self, "split", _req_plan_split(self.split))
        object.__setattr__(self, "status", _req_case_status(self.status))
        if not isinstance(self.qualified, bool):
            raise ModelingError("campaign_case_qualified_bool_required")
        object.__setattr__(self, "timing_mode", _req_nonempty_str(self.timing_mode, "campaign_case_timing_mode_required"))
        object.__setattr__(self, "capture_mode", _req_capture_mode(self.capture_mode))
        object.__setattr__(self, "clock_operator", _req_clock_operator(self.clock_operator))
        object.__setattr__(self, "residency", _req_residency(self.residency))
        object.__setattr__(self, "rank", _req_int(self.rank, "campaign_case_rank_required", minimum=0))
        ranks = tuple(_req_int(r, "campaign_case_ranks_invalid", minimum=0) for r in self.participating_ranks)
        if not ranks:
            raise ModelingError("campaign_case_participating_ranks_required")
        object.__setattr__(self, "participating_ranks", ranks)
        nodes = tuple(_req_nonempty_str(n, "campaign_case_timed_nodes_invalid") for n in self.timed_nodes)
        if not nodes:
            raise ModelingError("campaign_case_timed_nodes_required")
        object.__setattr__(self, "timed_nodes", nodes)
        p50 = _req_float(self.p50_us, "campaign_case_p50_invalid", strictly_positive=True)
        p99 = _req_float(self.p99_us, "campaign_case_p99_invalid", strictly_positive=True)
        object.__setattr__(self, "p50_us", p50)
        object.__setattr__(self, "p99_us", p99)
        object.__setattr__(self, "p90_us", _opt_float(self.p90_us, "campaign_case_p90_invalid", strictly_positive=True))
        object.__setattr__(self, "min_us", _opt_float(self.min_us, "campaign_case_min_invalid", strictly_positive=True))
        object.__setattr__(self, "iqr_us", _opt_float(self.iqr_us, "campaign_case_iqr_invalid", minimum=0.0))
        object.__setattr__(self, "cv_pct", _opt_float(self.cv_pct, "campaign_case_cv_invalid", minimum=0.0))
        object.__setattr__(self, "samples_count", _req_int(self.samples_count, "campaign_case_samples_invalid", minimum=1))
        object.__setattr__(self, "observed_clock_mhz", _req_float(self.observed_clock_mhz, "campaign_case_clock_invalid", strictly_positive=True))
        object.__setattr__(self, "hbm_peak_bytes", _opt_int(self.hbm_peak_bytes, "campaign_case_hbm_peak_invalid", minimum=0))
        object.__setattr__(self, "capture_uri", _req_nonempty_str(self.capture_uri, "campaign_case_uri_required"))
        object.__setattr__(self, "raw_sha256", _validate_sha256_hex(self.raw_sha256, "campaign_case_raw_sha256_invalid"))
        object.__setattr__(self, "actual_executed_shape", _freeze_json_mapping(self.actual_executed_shape, "campaign_case_shape_invalid"))
        extra = _freeze_json_mapping(self.extra_fields, "campaign_case_extra_fields_invalid")
        assert_no_forbidden_prediction_keys(extra, "$.cases[].extra_fields")
        for key_name in extra:
            if key_name in _CAMPAIGN_CASE_KNOWN_FIELDS:
                raise ModelingError(f"campaign_case_extra_field_collision:{key_name}")
        object.__setattr__(self, "extra_fields", extra)

    def to_dict(self) -> dict[str, Any]:
        return {
            **_plain_json(self.extra_fields),
            "acquisition_uid": self.acquisition_uid,
            "case_id": self.case_id,
            "canonical_key_digest": self.canonical_key_digest,
            "boundary": self.boundary,
            "split": self.split,
            "status": self.status,
            "qualified": self.qualified,
            "timing_mode": self.timing_mode,
            "capture_mode": self.capture_mode,
            "clock_operator": self.clock_operator,
            "residency": self.residency,
            "rank": self.rank,
            "participating_ranks": list(self.participating_ranks),
            "timed_nodes": list(self.timed_nodes),
            "p50_us": self.p50_us,
            "p90_us": self.p90_us,
            "p99_us": self.p99_us,
            "min_us": self.min_us,
            "iqr_us": self.iqr_us,
            "cv_pct": self.cv_pct,
            "samples_count": self.samples_count,
            "observed_clock_mhz": self.observed_clock_mhz,
            "hbm_peak_bytes": self.hbm_peak_bytes,
            "actual_executed_shape": _plain_json(self.actual_executed_shape),
            "cache_status": self.cache_status,
            "measured_in_run_id": self.measured_in_run_id,
            "run_shard": self.run_shard,
            "cold_coverage_verified": self.cold_coverage_verified,
            "traced_interval_verified": self.traced_interval_verified,
            "family": self.family,
            "geometry_key": self.geometry_key,
            "upstream_abi": self.upstream_abi,
            "abi_backend": self.abi_backend,
            "bench_builder": self.bench_builder,
            "capture_uri": self.capture_uri,
            "raw_sha256": self.raw_sha256,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CampaignCaseRecord:
        if not isinstance(raw, Mapping):
            raise ModelingError("campaign_case_requires_mapping")
        assert_no_forbidden_prediction_keys(raw, "$.cases[]")
        rank = _req_int(raw.get("rank", 0), "campaign_case_rank_required", minimum=0)
        raw_ranks = raw.get("participating_ranks")
        ranks = tuple(raw_ranks) if raw_ranks else (rank,)
        raw_nodes = raw.get("timed_nodes")
        nodes = tuple(raw_nodes) if raw_nodes else ("n0",)
        extra = {
            str(k): v for k, v in raw.items()
            if str(k) not in _CAMPAIGN_CASE_KNOWN_FIELDS
        }
        return cls(
            acquisition_uid=raw["acquisition_uid"],
            case_id=raw["case_id"],
            canonical_key_digest=raw["canonical_key_digest"],
            boundary=raw.get("boundary", "device_kernel"),
            split=raw["split"],
            status=raw["status"],
            qualified=raw.get("qualified", True),
            timing_mode=raw["timing_mode"],
            capture_mode=raw.get("capture_mode", "aclgraph" if raw["timing_mode"] == "aclgraph" else "eager"),
            clock_operator=raw["clock_operator"],
            residency=raw.get("residency", "cold_hbm"),
            rank=rank,
            participating_ranks=ranks,
            timed_nodes=nodes,
            p50_us=raw["p50_us"],
            p90_us=raw.get("p90_us"),
            p99_us=raw["p99_us"],
            min_us=raw.get("min_us"),
            iqr_us=raw.get("iqr_us"),
            cv_pct=raw.get("cv_pct"),
            samples_count=raw["samples_count"],
            observed_clock_mhz=raw["observed_clock_mhz"],
            hbm_peak_bytes=raw.get("hbm_peak_bytes"),
            actual_executed_shape=raw.get("actual_executed_shape", {}),
            cache_status=str(raw.get("cache_status") or "measured"),
            measured_in_run_id=str(raw.get("measured_in_run_id") or ""),
            run_shard=str(raw.get("run_shard") or ""),
            cold_coverage_verified=bool(raw.get("cold_coverage_verified", True)),
            traced_interval_verified=bool(raw.get("traced_interval_verified", False)),
            family=str(raw.get("family") or ""),
            geometry_key=str(raw.get("geometry_key") or ""),
            upstream_abi=str(raw.get("upstream_abi") or ""),
            abi_backend=str(raw.get("abi_backend") or ""),
            bench_builder=str(raw.get("bench_builder") or ""),
            capture_uri=raw["capture_uri"],
            raw_sha256=raw["raw_sha256"],
            extra_fields=extra,
        )


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    """Frozen ``mindie-bench-campaign/1`` sealed silicon measurement manifest."""

    campaign_id: str
    env_fingerprint_id: str
    plan_id: str
    plan_sha256: str
    version_fingerprint: BenchEnvFingerprint
    clock_operator: ClockOperator
    observations_jsonl: str
    observations_sha256: str
    cases: tuple[CampaignCaseRecord, ...]
    blocked_cases: tuple[Mapping[str, Any], ...] = ()
    worker_diagnostics: tuple[Mapping[str, Any], ...] = ()
    run_shards: tuple[RunShardDescriptor, ...] = ()
    cache_partition: CachePartition = "shared"
    training_manifest_sha256: str | None = None
    campaign_sha256: str = ""
    schema: Literal["mindie-bench-campaign/1"] = BENCH_CAMPAIGN_V1_SCHEMA
    extra_fields: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.schema != BENCH_CAMPAIGN_V1_SCHEMA:
            raise ModelingError(f"campaign_schema_invalid:{self.schema!r}")
        object.__setattr__(self, "campaign_id", _req_nonempty_str(self.campaign_id, "campaign_id_required"))
        if not isinstance(self.version_fingerprint, BenchEnvFingerprint):
            raise ModelingError("campaign_version_fingerprint_invalid")
        env_id = _req_nonempty_str(self.env_fingerprint_id, "campaign_env_id_required")
        if env_id != self.version_fingerprint.env_fingerprint_id:
            raise ModelingError(f"campaign_env_id_mismatch:{env_id}!={self.version_fingerprint.env_fingerprint_id}")
        object.__setattr__(self, "env_fingerprint_id", env_id)
        object.__setattr__(self, "plan_id", _req_nonempty_str(self.plan_id, "campaign_plan_id_required"))
        object.__setattr__(self, "plan_sha256", _validate_sha256_hex(self.plan_sha256, "campaign_plan_sha256_invalid"))
        object.__setattr__(self, "clock_operator", _req_clock_operator(self.clock_operator))
        object.__setattr__(self, "observations_jsonl", _req_nonempty_str(self.observations_jsonl, "campaign_obs_jsonl_required"))
        object.__setattr__(self, "observations_sha256", _validate_sha256_hex(self.observations_sha256, "campaign_obs_sha256_invalid"))
        object.__setattr__(self, "cache_partition", _req_cache_partition(self.cache_partition))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "blocked_cases", tuple(_freeze_json_mapping(b, "campaign_blocked_invalid") for b in self.blocked_cases))
        object.__setattr__(self, "worker_diagnostics", tuple(_freeze_json_mapping(d, "campaign_diag_invalid") for d in self.worker_diagnostics))
        object.__setattr__(self, "run_shards", tuple(self.run_shards))
        if self.training_manifest_sha256 is not None:
            object.__setattr__(
                self, "training_manifest_sha256",
                _validate_sha256_hex(self.training_manifest_sha256, "campaign_training_manifest_sha256_invalid"),
            )
        if self.campaign_sha256:
            _validate_sha256_hex(self.campaign_sha256, "campaign_sha256_invalid")
        extra = _freeze_json_mapping(self.extra_fields, "campaign_extra_fields_invalid")
        assert_no_forbidden_prediction_keys(extra, "$.extra_fields")
        for key_name in extra:
            if key_name in _CAMPAIGN_KNOWN_FIELDS:
                raise ModelingError(f"campaign_extra_field_collision:{key_name}")
        object.__setattr__(self, "extra_fields", extra)

    def to_dict(self, *, include_seal: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **_plain_json(self.extra_fields),
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "env_fingerprint_id": self.env_fingerprint_id,
            "cache_partition": self.cache_partition,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "version_fingerprint": self.version_fingerprint.to_dict(),
            "clock_operator": self.clock_operator,
            "observations_jsonl": self.observations_jsonl,
            "observations_sha256": self.observations_sha256,
            "run_shards": [s.to_dict() for s in self.run_shards],
            "training_manifest_sha256": self.training_manifest_sha256,
            "cases": [c.to_dict() for c in self.cases],
            "blocked_cases": [_plain_json(b) for b in self.blocked_cases],
            "worker_diagnostics": [_plain_json(d) for d in self.worker_diagnostics],
        }
        if include_seal:
            payload["campaign_sha256"] = self.campaign_sha256 or compute_campaign_sha256(payload)
        return payload

    def with_computed_seal(self) -> CampaignManifest:
        seal = compute_campaign_sha256(self.to_dict(include_seal=False))
        return CampaignManifest(
            campaign_id=self.campaign_id,
            env_fingerprint_id=self.env_fingerprint_id,
            plan_id=self.plan_id,
            plan_sha256=self.plan_sha256,
            version_fingerprint=self.version_fingerprint,
            clock_operator=self.clock_operator,
            observations_jsonl=self.observations_jsonl,
            observations_sha256=self.observations_sha256,
            cases=self.cases,
            blocked_cases=self.blocked_cases,
            worker_diagnostics=self.worker_diagnostics,
            run_shards=self.run_shards,
            cache_partition=self.cache_partition,
            training_manifest_sha256=self.training_manifest_sha256,
            campaign_sha256=seal,
            schema=self.schema,
            extra_fields=self.extra_fields,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, verify_seal: bool = True) -> CampaignManifest:
        if not isinstance(raw, Mapping):
            raise ModelingError("campaign_manifest_requires_mapping")
        assert_no_forbidden_prediction_keys(raw, "$")
        if raw.get("schema") != BENCH_CAMPAIGN_V1_SCHEMA:
            raise ModelingError(f"campaign_schema_invalid:{raw.get('schema')!r}")
        if verify_seal:
            recorded = _validate_sha256_hex(raw.get("campaign_sha256"), "campaign_sha256_missing")
            expected = compute_campaign_sha256(raw)
            if recorded != expected:
                raise ModelingError(f"campaign_sha256_mismatch:{recorded}!={expected}")
        env_fp = BenchEnvFingerprint.from_mapping(raw["version_fingerprint"])
        cases = tuple(CampaignCaseRecord.from_mapping(c) for c in raw.get("cases", ()))
        shards = tuple(RunShardDescriptor.from_mapping(s) for s in raw.get("run_shards", ()))
        extra = {
            str(k): v for k, v in raw.items()
            if str(k) not in _CAMPAIGN_KNOWN_FIELDS
        }
        return cls(
            campaign_id=raw["campaign_id"],
            env_fingerprint_id=raw["env_fingerprint_id"],
            plan_id=raw["plan_id"],
            plan_sha256=raw["plan_sha256"],
            version_fingerprint=env_fp,
            clock_operator=raw["clock_operator"],
            observations_jsonl=raw["observations_jsonl"],
            observations_sha256=raw["observations_sha256"],
            cases=cases,
            blocked_cases=tuple(raw.get("blocked_cases", ())),
            worker_diagnostics=tuple(raw.get("worker_diagnostics", ())),
            run_shards=shards,
            cache_partition=raw.get("cache_partition", env_fp.cache_partition),
            training_manifest_sha256=raw.get("training_manifest_sha256"),
            campaign_sha256=str(raw.get("campaign_sha256") or ""),
            schema=BENCH_CAMPAIGN_V1_SCHEMA,
            extra_fields=extra,
        )


@dataclass(frozen=True, slots=True)
class HardwareNominalSpec:
    """Un-derated public datasheet hardware nominal specs from ``profiles/ascend.json``.

    Strictly ``utilization == 1.0`` and ``launch_us == 0.0``. Never populated
    from ``PrimitiveTable`` sustained rates or link framing deratings.
    """

    sku: str
    soc_arch: SocArch
    aic_count: int
    aiv_count: int
    hbm_capacity_bytes: int
    l2_capacity_bytes: int
    ub_capacity_bytes: int
    nominal_clock_mhz: float
    cube_peaks_flops_per_s: Mapping[str, float]
    vector_peak_ops_per_s: float
    hbm_peak_bytes_per_s: float
    raw_link_peaks_bytes_per_s: Mapping[str, float]
    utilization: float = NOMINAL_UTILIZATION
    launch_us: float = NOMINAL_LAUNCH_US

    def __post_init__(self) -> None:
        object.__setattr__(self, "sku", _req_nonempty_str(self.sku, "nominal_spec_sku_required"))
        object.__setattr__(self, "soc_arch", _req_soc_arch(self.soc_arch, "nominal_spec_soc_arch_invalid"))
        object.__setattr__(self, "aic_count", _req_int(self.aic_count, "nominal_spec_aic_invalid", minimum=1))
        object.__setattr__(self, "aiv_count", _req_int(self.aiv_count, "nominal_spec_aiv_invalid", minimum=1))
        object.__setattr__(self, "hbm_capacity_bytes", _req_int(self.hbm_capacity_bytes, "nominal_spec_hbm_cap_invalid", minimum=1))
        object.__setattr__(self, "l2_capacity_bytes", _req_int(self.l2_capacity_bytes, "nominal_spec_l2_cap_invalid", minimum=1))
        object.__setattr__(self, "ub_capacity_bytes", _req_int(self.ub_capacity_bytes, "nominal_spec_ub_cap_invalid", minimum=1))
        object.__setattr__(self, "nominal_clock_mhz", _req_float(self.nominal_clock_mhz, "nominal_spec_clock_invalid", strictly_positive=True))
        peaks = {
            _req_nonempty_str(str(k), "nominal_spec_peak_dtype_invalid").lower(): _req_float(v, f"nominal_spec_peak_{k}", strictly_positive=True)
            for k, v in sorted(self.cube_peaks_flops_per_s.items(), key=lambda kv: str(kv[0]))
        }
        if not peaks:
            raise ModelingError("nominal_spec_cube_peaks_required")
        object.__setattr__(self, "cube_peaks_flops_per_s", MappingProxyType(peaks))
        object.__setattr__(self, "vector_peak_ops_per_s", _req_float(self.vector_peak_ops_per_s, "nominal_spec_vec_peak_invalid", strictly_positive=True))
        object.__setattr__(self, "hbm_peak_bytes_per_s", _req_float(self.hbm_peak_bytes_per_s, "nominal_spec_hbm_peak_invalid", strictly_positive=True))
        links = {
            _req_nonempty_str(str(k), "nominal_spec_link_tier_invalid"): _req_float(v, f"nominal_spec_link_{k}", strictly_positive=True)
            for k, v in sorted(self.raw_link_peaks_bytes_per_s.items(), key=lambda kv: str(kv[0]))
        }
        object.__setattr__(self, "raw_link_peaks_bytes_per_s", MappingProxyType(links))
        util = _req_float(self.utilization, "nominal_spec_utilization_invalid", strictly_positive=True)
        launch = _req_float(self.launch_us, "nominal_spec_launch_invalid", minimum=0.0)
        if util != NOMINAL_UTILIZATION:
            raise ModelingError(f"nominal_spec_must_be_underated:utilization={util}")
        if launch != NOMINAL_LAUNCH_US:
            raise ModelingError(f"nominal_spec_must_have_zero_launch:launch_us={launch}")
        object.__setattr__(self, "utilization", util)
        object.__setattr__(self, "launch_us", launch)

    def cube_peak_for_dtype(self, dtype: str) -> float:
        low = str(dtype or "bf16").strip().lower()
        for cand in _DTYPE_PEAK_ALIASES.get(low, (low,)):
            if cand in self.cube_peaks_flops_per_s:
                return self.cube_peaks_flops_per_s[cand]
        raise ModelingError(f"nominal_spec_unsupported_cube_dtype:{self.sku}:{dtype}")

    def link_peak_for_tier(self, tier: str = "d2d_intra_node") -> float:
        if tier in self.raw_link_peaks_bytes_per_s:
            return self.raw_link_peaks_bytes_per_s[tier]
        mapped = _LINK_TIER_CHANNEL_MAP.get(str(tier or "").strip().lower())
        if mapped is not None and mapped in self.raw_link_peaks_bytes_per_s:
            return self.raw_link_peaks_bytes_per_s[mapped]
        raise ModelingError(f"nominal_spec_missing_link_tier:{self.sku}:{tier}")

    def nominal_theo_us(
        self,
        *,
        cube_flops: float = 0.0,
        cube_dtype: str = "bf16",
        vector_ops: float = 0.0,
        hbm_read_bytes: float = 0.0,
        hbm_write_bytes: float = 0.0,
        link_bytes: float = 0.0,
        link_tier: str = "d2d_intra_node",
    ) -> float:
        """Compute un-derated physical roofline lower bound (us)."""
        t_cube = (cube_flops / self.cube_peak_for_dtype(cube_dtype)) if cube_flops > 0.0 else 0.0
        t_vec = (vector_ops / self.vector_peak_ops_per_s) if vector_ops > 0.0 else 0.0
        hbm_total = hbm_read_bytes + hbm_write_bytes
        t_hbm = (hbm_total / self.hbm_peak_bytes_per_s) if hbm_total > 0.0 else 0.0
        t_link = (link_bytes / self.link_peak_for_tier(link_tier)) if link_bytes > 0.0 else 0.0
        return max(t_cube, t_vec, t_hbm, t_link) * US_PER_SECOND

    def utilization_pcts(
        self,
        duration_us: float,
        *,
        cube_flops: float = 0.0,
        cube_dtype: str = "bf16",
        hbm_read_bytes: float = 0.0,
        hbm_write_bytes: float = 0.0,
        link_bytes: float = 0.0,
        link_tier: str = "d2d_intra_node",
    ) -> tuple[float, float, float, float]:
        """Return ``(mfu_pct, mbu_rw_pct, mbu_read_pct, link_bu_pct)`` vs nominal peaks."""
        dur_s = _req_float(duration_us, "utilization_duration_us_invalid", strictly_positive=True) / US_PER_SECOND
        mfu = (
            (cube_flops / (dur_s * self.cube_peak_for_dtype(cube_dtype))) * PCT_SCALE
            if cube_flops > 0.0 else 0.0
        )
        hbm_denom = dur_s * self.hbm_peak_bytes_per_s
        mbu_rw = ((hbm_read_bytes + hbm_write_bytes) / hbm_denom) * PCT_SCALE if (hbm_read_bytes + hbm_write_bytes) > 0.0 else 0.0
        mbu_read = (hbm_read_bytes / hbm_denom) * PCT_SCALE if hbm_read_bytes > 0.0 else 0.0
        link_bu = (
            (link_bytes / (dur_s * self.link_peak_for_tier(link_tier))) * PCT_SCALE
            if link_bytes > 0.0 else 0.0
        )
        return mfu, mbu_rw, mbu_read, link_bu

    @classmethod
    def from_ascend_profile(cls, sku: str, profile: Mapping[str, Any]) -> HardwareNominalSpec:
        """Build strictly un-derated nominal spec from one ``profiles/ascend.json`` entry."""
        device = profile.get("device")
        if not isinstance(device, Mapping):
            raise ModelingError(f"nominal_spec_profile_missing_device:{sku}")
        rates = device.get("rates")
        peaks = device.get("compute_peaks")
        buffers = device.get("buffers")
        res_units = device.get("resource_units")
        if not (isinstance(rates, Mapping) and isinstance(peaks, Mapping) and isinstance(buffers, Mapping)):
            raise ModelingError(f"nominal_spec_profile_incomplete:{sku}")
        decl = (profile.get("source") or {}).get("declaration") if isinstance(profile.get("source"), Mapping) else None
        clock_mhz = (decl or {}).get("configured_clock_mhz") if isinstance(decl, Mapping) else None
        if clock_mhz is None:
            raise ModelingError(f"nominal_spec_missing_configured_clock:{sku}")
        raw_links: dict[str, float] = {}
        channels = profile.get("transfer_channels")
        if isinstance(channels, Mapping):
            for ch_name, ch_cfg in channels.items():
                if not isinstance(ch_cfg, Mapping):
                    continue
                # STRICTLY raw un-derated wire peak (NEVER effective_* or calibrated_*).
                uni_peak = ch_cfg.get("uni_peak_bw_gbps")
                if isinstance(uni_peak, (int, float)) and not isinstance(uni_peak, bool) and uni_peak > 0.0:
                    raw_links[str(ch_name)] = float(uni_peak) * BYTES_PER_GB
        aiv_units = res_units.get("vector") if isinstance(res_units, Mapping) else None
        return cls(
            sku=sku,
            soc_arch=profile["architecture"],
            aic_count=device["compute_units"],
            aiv_count=aiv_units if aiv_units is not None else int(device["compute_units"]) * 2,
            hbm_capacity_bytes=device["memory_bytes"],
            l2_capacity_bytes=buffers["l2"],
            ub_capacity_bytes=buffers["ub"],
            nominal_clock_mhz=clock_mhz,
            cube_peaks_flops_per_s={str(k): float(v) for k, v in peaks.items()},
            vector_peak_ops_per_s=rates["vector"],
            hbm_peak_bytes_per_s=rates["hbm"],
            raw_link_peaks_bytes_per_s=raw_links,
            utilization=NOMINAL_UTILIZATION,
            launch_us=NOMINAL_LAUNCH_US,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "aic_count": self.aic_count,
            "aiv_count": self.aiv_count,
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "l2_capacity_bytes": self.l2_capacity_bytes,
            "ub_capacity_bytes": self.ub_capacity_bytes,
            "nominal_clock_mhz": self.nominal_clock_mhz,
            "cube_peaks_flops_per_s": dict(sorted(self.cube_peaks_flops_per_s.items())),
            "vector_peak_ops_per_s": self.vector_peak_ops_per_s,
            "hbm_peak_bytes_per_s": self.hbm_peak_bytes_per_s,
            "raw_link_peaks_bytes_per_s": dict(sorted(self.raw_link_peaks_bytes_per_s.items())),
            "utilization": self.utilization,
            "launch_us": self.launch_us,
        }


@dataclass(frozen=True, slots=True)
class HbmAdmissibilityReceipt:
    """Dual-peak per-die HBM + optional HiSparse host DDR admissibility receipt."""

    admissible: bool
    reason: str
    usable_hbm_bytes_per_die: int
    runtime_peak_hbm_bytes_by_die: tuple[int, ...]
    load_peak_hbm_bytes_by_die: tuple[int, ...]
    host_ddr_bytes: int = 0
    usable_host_ddr_bytes: int | None = None
    max_admissible_concurrency: int = 0
    evaluated_concurrency: int = 1
    cell_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.admissible, bool):
            raise ModelingError("hbm_receipt_admissible_bool_required")
        object.__setattr__(self, "reason", _req_nonempty_str(self.reason, "hbm_receipt_reason_required"))
        usable = _req_int(self.usable_hbm_bytes_per_die, "hbm_receipt_usable_invalid", minimum=1)
        rt_peaks = tuple(_req_int(v, "hbm_receipt_runtime_peak_invalid", minimum=0) for v in self.runtime_peak_hbm_bytes_by_die)
        ld_peaks = tuple(_req_int(v, "hbm_receipt_load_peak_invalid", minimum=0) for v in self.load_peak_hbm_bytes_by_die)
        if not rt_peaks or len(rt_peaks) != len(ld_peaks):
            raise ModelingError("hbm_receipt_die_peaks_mismatch")
        host_b = _req_int(self.host_ddr_bytes, "hbm_receipt_host_ddr_invalid", minimum=0)
        usable_host = _opt_int(self.usable_host_ddr_bytes, "hbm_receipt_usable_host_ddr_invalid", minimum=0)
        object.__setattr__(self, "usable_hbm_bytes_per_die", usable)
        object.__setattr__(self, "runtime_peak_hbm_bytes_by_die", rt_peaks)
        object.__setattr__(self, "load_peak_hbm_bytes_by_die", ld_peaks)
        object.__setattr__(self, "host_ddr_bytes", host_b)
        object.__setattr__(self, "usable_host_ddr_bytes", usable_host)
        object.__setattr__(self, "max_admissible_concurrency", _req_int(self.max_admissible_concurrency, "hbm_receipt_max_conc_invalid", minimum=0))
        object.__setattr__(self, "evaluated_concurrency", _req_int(self.evaluated_concurrency, "hbm_receipt_eval_conc_invalid", minimum=1))
        if self.admissible:
            if any(p > usable for p in rt_peaks) or any(p > usable for p in ld_peaks):
                raise ModelingError("hbm_admissibility_dual_peak_violation")
            if usable_host is not None and host_b > usable_host:
                raise ModelingError("hbm_admissibility_host_ddr_violation")

    @property
    def max_runtime_peak_hbm_bytes(self) -> int:
        return max(self.runtime_peak_hbm_bytes_by_die)

    @property
    def max_load_peak_hbm_bytes(self) -> int:
        return max(self.load_peak_hbm_bytes_by_die)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "admissible": self.admissible,
            "reason": self.reason,
            "usable_hbm_bytes_per_die": self.usable_hbm_bytes_per_die,
            "runtime_peak_hbm_bytes_by_die": list(self.runtime_peak_hbm_bytes_by_die),
            "load_peak_hbm_bytes_by_die": list(self.load_peak_hbm_bytes_by_die),
            "max_runtime_peak_hbm_bytes": self.max_runtime_peak_hbm_bytes,
            "max_load_peak_hbm_bytes": self.max_load_peak_hbm_bytes,
            "host_ddr_bytes": self.host_ddr_bytes,
            "usable_host_ddr_bytes": self.usable_host_ddr_bytes,
            "max_admissible_concurrency": self.max_admissible_concurrency,
            "evaluated_concurrency": self.evaluated_concurrency,
        }


@dataclass(frozen=True, slots=True)
class SeqCliffPoint:
    """Physics-derived sequence-length probe point (AG014-clean)."""

    seq_len: int
    cliff_kind: CliffKind
    split: PlanSplit
    priority: int
    formula_expr: str
    hw_inputs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "seq_len", _req_int(self.seq_len, "seq_cliff_seq_len_invalid", minimum=1))
        object.__setattr__(self, "cliff_kind", _req_cliff_kind(self.cliff_kind))
        object.__setattr__(self, "split", _req_plan_split(self.split))
        object.__setattr__(self, "priority", _req_int(self.priority, "seq_cliff_priority_invalid", minimum=0))
        object.__setattr__(self, "formula_expr", _req_nonempty_str(self.formula_expr, "seq_cliff_formula_required"))
        object.__setattr__(self, "hw_inputs", _freeze_json_mapping(self.hw_inputs, "seq_cliff_hw_inputs_invalid"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "cliff_kind": self.cliff_kind,
            "split": self.split,
            "priority": self.priority,
            "formula_expr": self.formula_expr,
            "hw_inputs": _plain_json(self.hw_inputs),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ParallelArmSpec:
    """Declarative parallel-arm candidate (pure routed-expert EP default)."""

    arm_id: str
    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1
    dcp: int = 1
    pcp: int = 1
    sp: bool = False
    num_devices: int | None = None
    q_proj_tp: int | None = None
    o_proj_tp: int | None = None
    dcp_linear_tp: int | None = None
    shared_expert_tp: int | None = None
    routed_expert_tp: int = 1
    sparse_kv_offload: bool = False
    deployment_mode: str = "colocated"
    auto_sla: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _req_nonempty_str(self.arm_id, "parallel_arm_id_required"))
        for fname in ("dp", "tp", "ep", "pp", "dcp", "pcp", "routed_expert_tp"):
            object.__setattr__(self, fname, _req_int(getattr(self, fname), f"parallel_arm_{fname}_invalid", minimum=1))
        for fname in ("num_devices", "q_proj_tp", "o_proj_tp", "dcp_linear_tp", "shared_expert_tp"):
            object.__setattr__(self, fname, _opt_int(getattr(self, fname), f"parallel_arm_{fname}_invalid", minimum=1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "dp": self.dp,
            "tp": self.tp,
            "ep": self.ep,
            "pp": self.pp,
            "dcp": self.dcp,
            "pcp": self.pcp,
            "sp": self.sp,
            "num_devices": self.num_devices,
            "q_proj_tp": self.q_proj_tp,
            "o_proj_tp": self.o_proj_tp,
            "dcp_linear_tp": self.dcp_linear_tp,
            "shared_expert_tp": self.shared_expert_tp,
            "routed_expert_tp": self.routed_expert_tp,
            "sparse_kv_offload": self.sparse_kv_offload,
            "deployment_mode": self.deployment_mode,
            "auto_sla": self.auto_sla,
        }


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    """Declarative input specification for ``mindie-sim matrix`` sweeps."""

    models: tuple[str, ...]
    skus: tuple[str, ...]
    phases: tuple[PhaseKind, ...] = ("prefill", "decode")
    seq_min: int = DEFAULT_SEQ_MIN
    seq_max: int = DEFAULT_SEQ_MAX
    anchor_ladder: tuple[int, ...] = DEFAULT_ANCHOR_LADDER
    auto_physical_cliffs: bool = True
    cliff_budget_per_arm: int = DEFAULT_CLIFF_BUDGET_PER_ARM
    concurrencies: tuple[int, ...] = (1,)
    parallel_arms: tuple[ParallelArmSpec, ...] = ()
    stages: tuple[str, ...] = ("before",)
    quant_policy: str = "auto_va"
    mtp_proposal_tokens: int = 0
    mtp_acceptance_length: float | None = None
    deployment_mode: str = "colocated"
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    cache_partition: CachePartition = "shared"

    def __post_init__(self) -> None:
        models = tuple(_req_nonempty_str(m, "matrix_spec_model_required") for m in self.models)
        skus = tuple(_req_nonempty_str(s, "matrix_spec_sku_required") for s in self.skus)
        if not models or not skus:
            raise ModelingError("matrix_spec_models_and_skus_required")
        phases = tuple(_req_phase(p) for p in self.phases)
        if not phases:
            raise ModelingError("matrix_spec_phases_required")
        seq_min = _req_int(self.seq_min, "matrix_spec_seq_min_invalid", minimum=1)
        seq_max = _req_int(self.seq_max, "matrix_spec_seq_max_invalid", minimum=seq_min)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "skus", skus)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "seq_min", seq_min)
        object.__setattr__(self, "seq_max", seq_max)
        object.__setattr__(self, "anchor_ladder", tuple(_req_int(v, "matrix_spec_anchor_invalid", minimum=1) for v in self.anchor_ladder))
        object.__setattr__(self, "cliff_budget_per_arm", _req_int(self.cliff_budget_per_arm, "matrix_spec_cliff_budget_invalid", minimum=0))
        object.__setattr__(self, "concurrencies", tuple(_req_int(v, "matrix_spec_concurrency_invalid", minimum=1) for v in self.concurrencies))
        object.__setattr__(self, "parallel_arms", tuple(self.parallel_arms))
        object.__setattr__(self, "stages", tuple(_req_nonempty_str(s, "matrix_spec_stage_invalid") for s in self.stages))
        object.__setattr__(self, "mtp_proposal_tokens", _req_int(self.mtp_proposal_tokens, "matrix_spec_mtp_invalid", minimum=0))
        object.__setattr__(self, "mtp_acceptance_length", _opt_float(self.mtp_acceptance_length, "matrix_spec_mtp_accept_invalid", strictly_positive=True))
        object.__setattr__(self, "ttft_slo_ms", _opt_float(self.ttft_slo_ms, "matrix_spec_ttft_invalid", strictly_positive=True))
        object.__setattr__(self, "tpot_slo_ms", _opt_float(self.tpot_slo_ms, "matrix_spec_tpot_invalid", strictly_positive=True))
        object.__setattr__(self, "cache_partition", _req_cache_partition(self.cache_partition))

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "skus": list(self.skus),
            "phases": list(self.phases),
            "seq_min": self.seq_min,
            "seq_max": self.seq_max,
            "anchor_ladder": list(self.anchor_ladder),
            "auto_physical_cliffs": self.auto_physical_cliffs,
            "cliff_budget_per_arm": self.cliff_budget_per_arm,
            "concurrencies": list(self.concurrencies),
            "parallel_arms": [a.to_dict() for a in self.parallel_arms],
            "stages": list(self.stages),
            "quant_policy": self.quant_policy,
            "mtp_proposal_tokens": self.mtp_proposal_tokens,
            "mtp_acceptance_length": self.mtp_acceptance_length,
            "deployment_mode": self.deployment_mode,
            "ttft_slo_ms": self.ttft_slo_ms,
            "tpot_slo_ms": self.tpot_slo_ms,
            "cache_partition": self.cache_partition,
        }


@dataclass(frozen=True, slots=True)
class OperatorMatrixRow:
    """Single operator breakdown row in ``OperatorMatrixManifest``."""

    cell_id: str
    model_id: str
    sku: str
    soc_arch: SocArch
    phase: PhaseKind
    seq_len: int
    seq_point_kind: CliffKind
    concurrency: int
    parallel_arm: str
    layer_archetype_id: str
    native_layer: int
    model_scope: ModelScopeKind
    replica: int
    rank: int | None
    op_id: str
    occurrence_count: int
    family: str
    recipe: str
    abi_backend: str
    upstream_abi: str
    bench_builder: str
    bench_case_id: str
    capture_mode: CaptureMode
    formula_symbolic: str
    formula_terms: Mapping[str, Any]
    nominal_theo_us: float
    per_call_stage0_us: float
    model_exposed_stage0_us: float
    step_share_pct: float
    l_star_us: float
    hbm_read_bytes: int
    hbm_write_bytes: int
    sim_mfu_pct: float
    sim_mbu_rw_pct: float
    sim_mbu_read_pct: float
    sim_link_bu_pct: float = 0.0
    op_name: str = ""
    op_type: str = ""
    kernel_name: str = ""
    accelerator_core: AcceleratorCore = "AI_CORE"
    block_dim: int = 1
    mix_block_dim: int = 0
    stream_id: int = 0
    input_shapes: str = ""
    input_data_types: str = ""
    input_formats: str = ""
    output_shapes: str = ""
    output_data_types: str = ""
    output_formats: str = ""
    arithmetic_intensity: float = 0.0
    bound_type: BoundType = "Memory Bound (HBM)"
    stage1_us: float | None = None
    stage2_us: float | None = None
    stage3_us: float | None = None
    cube_flops: int = 0
    cube_dtype: str = "bf16"
    vector_ops: int = 0
    l2_bytes: int = 0
    ub_bytes: int = 0
    link_bytes: int = 0
    link_tier: str = "d2d_intra_node"
    bottleneck_resource: str = "hbm"
    tensor_shapes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    strides: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: MappingProxyType({}))
    dtypes: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    layouts: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    measured_us: float | None = None
    measured_p99_us: float | None = None
    measured_mfu_pct: float | None = None
    measured_mbu_rw_pct: float | None = None
    measured_mbu_read_pct: float | None = None
    measured_link_bu_pct: float | None = None
    ape_pct: float | None = None
    within_tolerance: bool | None = None
    l_star_inverted: bool | None = None
    env_fingerprint_id: str | None = None
    capture_uri: str | None = None
    stage: str = "before"
    clock_operator: ClockOperator = "elapsed_span"
    normalized_clock_mhz: float | None = None
    participating_ranks: tuple[int, ...] = ()
    timed_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _req_nonempty_str(self.cell_id, "matrix_row_cell_id_required"))
        object.__setattr__(self, "sku", _req_nonempty_str(self.sku, "matrix_row_sku_required"))
        object.__setattr__(self, "soc_arch", _req_soc_arch(self.soc_arch, "matrix_row_soc_arch_invalid"))
        object.__setattr__(self, "phase", _req_phase(self.phase))
        object.__setattr__(self, "seq_len", _req_int(self.seq_len, "matrix_row_seq_len_invalid", minimum=1))
        object.__setattr__(self, "seq_point_kind", _req_cliff_kind(self.seq_point_kind))
        object.__setattr__(self, "concurrency", _req_int(self.concurrency, "matrix_row_concurrency_invalid", minimum=1))
        object.__setattr__(self, "native_layer", _req_int(self.native_layer, "matrix_row_native_layer_invalid", minimum=0))
        object.__setattr__(self, "model_scope", _req_model_scope(self.model_scope))
        object.__setattr__(self, "replica", _req_int(self.replica, "matrix_row_replica_invalid", minimum=0))
        object.__setattr__(self, "rank", _opt_int(self.rank, "matrix_row_rank_invalid", minimum=0))
        object.__setattr__(self, "op_id", _req_nonempty_str(self.op_id, "matrix_row_op_id_required"))
        object.__setattr__(self, "occurrence_count", _req_int(self.occurrence_count, "matrix_row_occ_invalid", minimum=1))
        object.__setattr__(self, "capture_mode", _req_capture_mode(self.capture_mode))
        object.__setattr__(self, "formula_symbolic", _req_nonempty_str(self.formula_symbolic, "matrix_row_formula_required"))
        object.__setattr__(self, "formula_terms", _freeze_json_mapping(self.formula_terms, "matrix_row_formula_terms_invalid"))
        object.__setattr__(self, "nominal_theo_us", _req_float(self.nominal_theo_us, "matrix_row_nominal_theo_invalid", minimum=0.0))
        object.__setattr__(self, "per_call_stage0_us", _req_float(self.per_call_stage0_us, "matrix_row_stage0_invalid", minimum=0.0))
        object.__setattr__(self, "model_exposed_stage0_us", _req_float(self.model_exposed_stage0_us, "matrix_row_exposed_invalid", minimum=0.0))
        object.__setattr__(self, "step_share_pct", _req_float(self.step_share_pct, "matrix_row_share_invalid", minimum=0.0))
        object.__setattr__(self, "l_star_us", _req_float(self.l_star_us, "matrix_row_lstar_invalid", minimum=0.0))
        object.__setattr__(self, "hbm_read_bytes", _req_int(self.hbm_read_bytes, "matrix_row_hbm_read_invalid", minimum=0))
        object.__setattr__(self, "hbm_write_bytes", _req_int(self.hbm_write_bytes, "matrix_row_hbm_write_invalid", minimum=0))
        object.__setattr__(self, "sim_mfu_pct", _req_float(self.sim_mfu_pct, "matrix_row_sim_mfu_invalid", minimum=0.0))
        object.__setattr__(self, "sim_mbu_rw_pct", _req_float(self.sim_mbu_rw_pct, "matrix_row_sim_mbu_rw_invalid", minimum=0.0))
        object.__setattr__(self, "sim_mbu_read_pct", _req_float(self.sim_mbu_read_pct, "matrix_row_sim_mbu_read_invalid", minimum=0.0))
        object.__setattr__(self, "sim_link_bu_pct", _req_float(self.sim_link_bu_pct, "matrix_row_sim_link_bu_invalid", minimum=0.0))
        object.__setattr__(self, "accelerator_core", _req_accelerator_core(self.accelerator_core))
        object.__setattr__(self, "block_dim", _req_int(self.block_dim, "matrix_row_block_dim_invalid", minimum=0))
        object.__setattr__(self, "mix_block_dim", _req_int(self.mix_block_dim, "matrix_row_mix_block_dim_invalid", minimum=0))
        object.__setattr__(self, "stream_id", _req_int(self.stream_id, "matrix_row_stream_id_invalid", minimum=0))
        object.__setattr__(self, "arithmetic_intensity", _req_float(self.arithmetic_intensity, "matrix_row_ai_invalid", minimum=0.0))
        object.__setattr__(self, "bound_type", _req_bound_type(self.bound_type))
        object.__setattr__(self, "stage1_us", _opt_float(self.stage1_us, "matrix_row_stage1_invalid", minimum=0.0))
        object.__setattr__(self, "stage2_us", _opt_float(self.stage2_us, "matrix_row_stage2_invalid", minimum=0.0))
        object.__setattr__(self, "stage3_us", _opt_float(self.stage3_us, "matrix_row_stage3_invalid", minimum=0.0))
        object.__setattr__(self, "cube_flops", _req_int(self.cube_flops, "matrix_row_flops_invalid", minimum=0))
        object.__setattr__(self, "vector_ops", _req_int(self.vector_ops, "matrix_row_vec_ops_invalid", minimum=0))
        object.__setattr__(self, "l2_bytes", _req_int(self.l2_bytes, "matrix_row_l2_invalid", minimum=0))
        object.__setattr__(self, "ub_bytes", _req_int(self.ub_bytes, "matrix_row_ub_invalid", minimum=0))
        object.__setattr__(self, "link_bytes", _req_int(self.link_bytes, "matrix_row_link_invalid", minimum=0))
        object.__setattr__(self, "tensor_shapes", _freeze_json_mapping(self.tensor_shapes, "matrix_row_shapes_invalid"))
        object.__setattr__(self, "strides", _freeze_strides_map(self.strides, "matrix_row_strides_invalid"))
        object.__setattr__(self, "dtypes", _freeze_str_map(self.dtypes, "matrix_row_dtypes_invalid"))
        object.__setattr__(self, "layouts", _freeze_str_map(self.layouts, "matrix_row_layouts_invalid"))
        object.__setattr__(self, "measured_us", _opt_float(self.measured_us, "matrix_row_measured_us_invalid", strictly_positive=True))
        object.__setattr__(self, "measured_p99_us", _opt_float(self.measured_p99_us, "matrix_row_measured_p99_invalid", strictly_positive=True))
        object.__setattr__(self, "measured_mfu_pct", _opt_float(self.measured_mfu_pct, "matrix_row_measured_mfu_invalid", minimum=0.0))
        object.__setattr__(self, "measured_mbu_rw_pct", _opt_float(self.measured_mbu_rw_pct, "matrix_row_measured_mbu_rw_invalid", minimum=0.0))
        object.__setattr__(self, "measured_mbu_read_pct", _opt_float(self.measured_mbu_read_pct, "matrix_row_measured_mbu_read_invalid", minimum=0.0))
        object.__setattr__(self, "measured_link_bu_pct", _opt_float(self.measured_link_bu_pct, "matrix_row_measured_link_bu_invalid", minimum=0.0))
        object.__setattr__(self, "ape_pct", _opt_float(self.ape_pct, "matrix_row_ape_invalid", minimum=0.0))
        object.__setattr__(self, "stage", _req_nonempty_str(self.stage, "matrix_row_stage_required"))
        object.__setattr__(self, "clock_operator", _req_clock_operator(self.clock_operator))
        object.__setattr__(self, "normalized_clock_mhz", _opt_float(self.normalized_clock_mhz, "matrix_row_clock_invalid", strictly_positive=True))
        object.__setattr__(
            self, "participating_ranks",
            tuple(_req_int(r, "matrix_row_participating_ranks_invalid", minimum=0) for r in self.participating_ranks),
        )
        object.__setattr__(
            self, "timed_nodes",
            tuple(_req_nonempty_str(n, "matrix_row_timed_nodes_invalid") for n in self.timed_nodes),
        )

    @property
    def hbm_total_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def sim_mbu_pct(self) -> float:
        """Physical HBM bus utilization alias (``Read + Write``)."""
        return self.sim_mbu_rw_pct

    @property
    def measured_mbu_pct(self) -> float | None:
        """Measured physical HBM bus utilization alias (``Read + Write``)."""
        return self.measured_mbu_rw_pct

    def occurrence_key(self, stage: str | None = None) -> tuple[str, str, str, int, int, int | None, str]:
        use_stage = self.stage if stage is None else _req_nonempty_str(stage, "matrix_row_stage_required")
        return (self.cell_id, use_stage, self.model_scope, self.native_layer, self.replica, self.rank, self.op_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "stage": self.stage,
            "model_id": self.model_id,
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "phase": self.phase,
            "seq_len": self.seq_len,
            "seq_point_kind": self.seq_point_kind,
            "concurrency": self.concurrency,
            "parallel_arm": self.parallel_arm,
            "layer_archetype_id": self.layer_archetype_id,
            "native_layer": self.native_layer,
            "model_scope": self.model_scope,
            "replica": self.replica,
            "rank": self.rank,
            "op_id": self.op_id,
            "occurrence_count": self.occurrence_count,
            "family": self.family,
            "recipe": self.recipe,
            "abi_backend": self.abi_backend,
            "upstream_abi": self.upstream_abi,
            "bench_builder": self.bench_builder,
            "bench_case_id": self.bench_case_id,
            "capture_mode": self.capture_mode,
            "clock_operator": self.clock_operator,
            "normalized_clock_mhz": self.normalized_clock_mhz,
            "participating_ranks": list(self.participating_ranks),
            "timed_nodes": list(self.timed_nodes),
            "formula_symbolic": self.formula_symbolic,
            "formula_terms": _plain_json(self.formula_terms),
            "nominal_theo_us": self.nominal_theo_us,
            "per_call_stage0_us": self.per_call_stage0_us,
            "model_exposed_stage0_us": self.model_exposed_stage0_us,
            "step_share_pct": self.step_share_pct,
            "stage1_us": self.stage1_us,
            "stage2_us": self.stage2_us,
            "stage3_us": self.stage3_us,
            "l_star_us": self.l_star_us,
            "cube_flops": self.cube_flops,
            "cube_dtype": self.cube_dtype,
            "vector_ops": self.vector_ops,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_total_bytes": self.hbm_total_bytes,
            "l2_bytes": self.l2_bytes,
            "ub_bytes": self.ub_bytes,
            "link_bytes": self.link_bytes,
            "link_tier": self.link_tier,
            "bottleneck_resource": self.bottleneck_resource,
            "sim_mfu_pct": self.sim_mfu_pct,
            "sim_mbu_pct": self.sim_mbu_pct,
            "sim_mbu_rw_pct": self.sim_mbu_rw_pct,
            "sim_mbu_read_pct": self.sim_mbu_read_pct,
            "sim_link_bu_pct": self.sim_link_bu_pct,
            "op_name": self.op_name,
            "op_type": self.op_type,
            "kernel_name": self.kernel_name,
            "accelerator_core": self.accelerator_core,
            "block_dim": self.block_dim,
            "mix_block_dim": self.mix_block_dim,
            "stream_id": self.stream_id,
            "input_shapes": self.input_shapes,
            "input_data_types": self.input_data_types,
            "input_formats": self.input_formats,
            "output_shapes": self.output_shapes,
            "output_data_types": self.output_data_types,
            "output_formats": self.output_formats,
            "arithmetic_intensity": self.arithmetic_intensity,
            "bound_type": self.bound_type,
            "tensor_shapes": _plain_json(self.tensor_shapes),
            "strides": {k: list(v) for k, v in sorted(self.strides.items())},
            "dtypes": dict(sorted(self.dtypes.items())),
            "layouts": dict(sorted(self.layouts.items())),
            "measured_us": self.measured_us,
            "measured_p99_us": self.measured_p99_us,
            "measured_mfu_pct": self.measured_mfu_pct,
            "measured_mbu_pct": self.measured_mbu_pct,
            "measured_mbu_rw_pct": self.measured_mbu_rw_pct,
            "measured_mbu_read_pct": self.measured_mbu_read_pct,
            "measured_link_bu_pct": self.measured_link_bu_pct,
            "ape_pct": self.ape_pct,
            "within_tolerance": self.within_tolerance,
            "l_star_inverted": self.l_star_inverted,
            "env_fingerprint_id": self.env_fingerprint_id,
            "capture_uri": self.capture_uri,
        }

    def to_outward_dict(self) -> dict[str, Any]:
        """Project clean outward ``msprof``-aligned row free of internal simulator tokens."""
        out: dict[str, Any] = {
            "model_id": self.model_id,
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "phase": self.phase,
            "seq_len": self.seq_len,
            "concurrency": self.concurrency,
            "parallel_arm": self.parallel_arm,
            "op_name": self.op_name or self.op_id,
            "op_type": self.op_type or self.family,
            "kernel_name": self.kernel_name or self.upstream_abi,
            "accelerator_core": self.accelerator_core,
            "block_dim": self.block_dim,
            "mix_block_dim": self.mix_block_dim,
            "stream_id": self.stream_id,
            "input_shapes": self.input_shapes,
            "input_data_types": self.input_data_types,
            "input_formats": self.input_formats,
            "output_shapes": self.output_shapes,
            "output_data_types": self.output_data_types,
            "output_formats": self.output_formats,
            "occurrence_count": self.occurrence_count,
            "formula_symbolic": self.formula_symbolic,
            "formula_terms": _plain_json(self.formula_terms),
            "cube_flops": self.cube_flops,
            "vector_ops": self.vector_ops,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_total_bytes": self.hbm_total_bytes,
            "link_bytes": self.link_bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "bound_type": self.bound_type,
            "nominal_theo_us": self.nominal_theo_us,
            "estimated_per_call_us": self.per_call_stage0_us,
            "estimated_exposed_us": self.model_exposed_stage0_us,
            "step_share_pct": self.step_share_pct,
            "sim_mfu_pct": self.sim_mfu_pct,
            "sim_mbu_rw_pct": self.sim_mbu_rw_pct,
            "sim_mbu_read_pct": self.sim_mbu_read_pct,
            "sim_link_bu_pct": self.sim_link_bu_pct,
            "measured_us": self.measured_us,
            "measured_p99_us": self.measured_p99_us,
            "measured_mfu_pct": self.measured_mfu_pct,
            "measured_mbu_rw_pct": self.measured_mbu_rw_pct,
            "measured_mbu_read_pct": self.measured_mbu_read_pct,
            "measured_link_bu_pct": self.measured_link_bu_pct,
            "ape_pct": self.ape_pct,
        }
        assert_no_outward_internal_tokens(canonical_json(out), context="OperatorMatrixRow.to_outward_dict")
        return out


@dataclass(frozen=True, slots=True)
class StepSummary:
    """Whole-step matrix summary exposing both algorithmic ``model_mbu_pct`` and physical ``hbm_bus_util_pct``."""

    cell_id: str
    model_id: str
    sku: str
    soc_arch: SocArch
    phase: PhaseKind
    seq_len: int
    concurrency: int
    parallel_arm: str
    num_dies: int
    admissible: bool
    sim_step_us: float
    l_star_step_us: float
    active_weight_bytes: int
    kv_read_bytes: int
    new_kv_write_bytes: int
    total_step_hbm_rw_bytes: int
    total_step_cube_flops: int
    model_mbu_pct: float
    hbm_bus_util_pct: float
    step_sim_mfu_pct: float
    total_step_link_bytes: int = 0
    step_sim_link_bu_pct: float = 0.0
    measured_step_us: float | None = None
    step_delta_us: float | None = None
    measured_model_mbu_pct: float | None = None
    measured_hbm_bus_util_pct: float | None = None
    step_measured_mfu_pct: float | None = None
    step_measured_link_bu_pct: float | None = None
    dag_replayed: bool = False
    cube_dtype: str = "bf16"
    link_tier: str = "d2d_intra_node"
    env_fingerprint_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _req_nonempty_str(self.cell_id, "step_summary_cell_id_required"))
        object.__setattr__(self, "sku", _req_nonempty_str(self.sku, "step_summary_sku_required"))
        object.__setattr__(self, "soc_arch", _req_soc_arch(self.soc_arch, "step_summary_soc_arch_invalid"))
        object.__setattr__(self, "phase", _req_phase(self.phase))
        object.__setattr__(self, "seq_len", _req_int(self.seq_len, "step_summary_seq_len_invalid", minimum=1))
        object.__setattr__(self, "concurrency", _req_int(self.concurrency, "step_summary_conc_invalid", minimum=1))
        object.__setattr__(self, "num_dies", _req_int(self.num_dies, "step_summary_num_dies_invalid", minimum=1))
        object.__setattr__(self, "sim_step_us", _req_float(self.sim_step_us, "step_summary_sim_step_invalid", minimum=0.0))
        object.__setattr__(self, "l_star_step_us", _req_float(self.l_star_step_us, "step_summary_lstar_invalid", minimum=0.0))
        for fname in (
            "active_weight_bytes", "kv_read_bytes", "new_kv_write_bytes",
            "total_step_hbm_rw_bytes", "total_step_cube_flops", "total_step_link_bytes",
        ):
            object.__setattr__(self, fname, _req_int(getattr(self, fname), f"step_summary_{fname}_invalid", minimum=0))
        for fname in ("model_mbu_pct", "hbm_bus_util_pct", "step_sim_mfu_pct", "step_sim_link_bu_pct"):
            object.__setattr__(self, fname, _req_float(getattr(self, fname), f"step_summary_{fname}_invalid", minimum=0.0))
        object.__setattr__(self, "measured_step_us", _opt_float(self.measured_step_us, "step_summary_meas_step_invalid", strictly_positive=True))
        object.__setattr__(self, "step_delta_us", _opt_float(self.step_delta_us, "step_summary_delta_invalid"))
        for fname in ("measured_model_mbu_pct", "measured_hbm_bus_util_pct", "step_measured_mfu_pct", "step_measured_link_bu_pct"):
            object.__setattr__(self, fname, _opt_float(getattr(self, fname), f"step_summary_{fname}_invalid", minimum=0.0))

    @property
    def irreducible_model_bytes(self) -> int:
        return self.active_weight_bytes + self.kv_read_bytes + self.new_kv_write_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "model_id": self.model_id,
            "sku": self.sku,
            "soc_arch": self.soc_arch,
            "phase": self.phase,
            "seq_len": self.seq_len,
            "concurrency": self.concurrency,
            "parallel_arm": self.parallel_arm,
            "num_dies": self.num_dies,
            "admissible": self.admissible,
            "sim_step_us": self.sim_step_us,
            "l_star_step_us": self.l_star_step_us,
            "active_weight_bytes": self.active_weight_bytes,
            "kv_read_bytes": self.kv_read_bytes,
            "new_kv_write_bytes": self.new_kv_write_bytes,
            "irreducible_model_bytes": self.irreducible_model_bytes,
            "total_step_hbm_rw_bytes": self.total_step_hbm_rw_bytes,
            "total_step_cube_flops": self.total_step_cube_flops,
            "total_step_link_bytes": self.total_step_link_bytes,
            "model_mbu_pct": self.model_mbu_pct,
            "hbm_bus_util_pct": self.hbm_bus_util_pct,
            "step_sim_mfu_pct": self.step_sim_mfu_pct,
            "step_sim_link_bu_pct": self.step_sim_link_bu_pct,
            "measured_step_us": self.measured_step_us,
            "step_delta_us": self.step_delta_us,
            "measured_model_mbu_pct": self.measured_model_mbu_pct,
            "measured_hbm_bus_util_pct": self.measured_hbm_bus_util_pct,
            "step_measured_mfu_pct": self.step_measured_mfu_pct,
            "step_measured_link_bu_pct": self.step_measured_link_bu_pct,
            "dag_replayed": self.dag_replayed,
            "cube_dtype": self.cube_dtype,
            "link_tier": self.link_tier,
            "env_fingerprint_id": self.env_fingerprint_id,
        }


@dataclass(frozen=True, slots=True)
class OperatorMatrixManifest:
    """Outward-facing ``mindie-sim.operator-matrix/1`` deliverable manifest."""

    spec: MatrixSpec
    nominal_specs: Mapping[str, HardwareNominalSpec]
    cliffs: tuple[SeqCliffPoint, ...]
    hbm_receipts: tuple[HbmAdmissibilityReceipt, ...]
    rows: tuple[OperatorMatrixRow, ...]
    step_summaries: tuple[StepSummary, ...]
    reverse_index: Mapping[str, tuple[ReverseIndexEntry, ...]]
    plan_sha256: str | None = None
    campaign_id: str | None = None
    env_fingerprint_id: str | None = None
    schema: Literal["mindie-sim.operator-matrix/1"] = OPERATOR_MATRIX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OPERATOR_MATRIX_SCHEMA:
            raise ModelingError(f"operator_matrix_schema_invalid:{self.schema!r}")
        if not isinstance(self.spec, MatrixSpec):
            raise ModelingError("operator_matrix_spec_invalid")
        nom = {str(k): v for k, v in sorted(self.nominal_specs.items())}
        for v in nom.values():
            if not isinstance(v, HardwareNominalSpec):
                raise ModelingError("operator_matrix_nominal_spec_invalid")
        object.__setattr__(self, "nominal_specs", MappingProxyType(nom))
        object.__setattr__(self, "cliffs", tuple(self.cliffs))
        object.__setattr__(self, "hbm_receipts", tuple(self.hbm_receipts))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "step_summaries", tuple(self.step_summaries))
        rev = {str(k): tuple(v) for k, v in sorted(self.reverse_index.items())}
        object.__setattr__(self, "reverse_index", MappingProxyType(rev))
        if self.plan_sha256 is not None:
            object.__setattr__(self, "plan_sha256", _validate_sha256_hex(self.plan_sha256, "operator_matrix_plan_sha256_invalid"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "spec": self.spec.to_dict(),
            "nominal_specs": {k: v.to_dict() for k, v in sorted(self.nominal_specs.items())},
            "cliffs": [c.to_dict() for c in self.cliffs],
            "hbm_receipts": [r.to_dict() for r in self.hbm_receipts],
            "rows": [r.to_dict() for r in self.rows],
            "step_summaries": [s.to_dict() for s in self.step_summaries],
            "reverse_index": {
                cid: [e.to_dict() for e in entries]
                for cid, entries in sorted(self.reverse_index.items())
            },
            "plan_sha256": self.plan_sha256,
            "campaign_id": self.campaign_id,
            "env_fingerprint_id": self.env_fingerprint_id,
        }


__all__ = [
    "BENCH_CAMPAIGN_V1_SCHEMA",
    "BENCH_ENV_V1_SCHEMA",
    "BENCH_OBSERVATION_V2_SCHEMA",
    "BENCH_PLAN_V2_SCHEMA",
    "CANONICAL_DIGEST_HEX_LEN",
    "CANONICAL_KERNEL_KEY_FIELDS",
    "CANONICAL_KEY_SIG_VERSION",
    "CASE_ID_PREFIX",
    "CASE_RAW_SHA256_EXCLUDED_FIELDS",
    "FABRIC_COLLECTIVE_ABI_MAP",
    "FORBIDDEN_CANONICAL_KEY_FIELDS",
    "FORBIDDEN_PREDICTION_KEYS",
    "OPERATOR_MATRIX_SCHEMA",
    "OUTWARD_FORBIDDEN_INTERNAL_TOKENS",
    "STATIC_ENV_BUCKET_FIELDS",
    "VALID_ACCELERATOR_CORES",
    "VALID_BOUND_TYPES",
    "AcceleratorCore",
    "BenchEnvFingerprint",
    "BenchObservationV2",
    "BenchPlanCaseV2",
    "BenchPlanV2",
    "BoundType",
    "CachePartition",
    "CampaignCaseRecord",
    "CampaignCaseStatus",
    "CampaignManifest",
    "CanonicalKernelKey",
    "CaptureMode",
    "CliffKind",
    "ClockOperator",
    "HardwareNominalSpec",
    "HbmAdmissibilityReceipt",
    "KernelKeyKind",
    "MatrixSpec",
    "ModelScopeKind",
    "OperatorMatrixManifest",
    "OperatorMatrixRow",
    "ParallelArmSpec",
    "PhaseKind",
    "PlanSplit",
    "ResidencyMode",
    "ReverseIndexEntry",
    "RunShardDescriptor",
    "SeqCliffPoint",
    "StepSummary",
    "assert_no_forbidden_prediction_keys",
    "assert_no_outward_internal_tokens",
    "canonical_case_id",
    "canonical_json",
    "canonical_key_digest",
    "canonical_sha256",
    "compute_campaign_sha256",
    "compute_case_raw_sha256",
    "compute_env_fingerprint_id",
    "compute_plan_sha256",
    "resolve_fabric_collective_abi",
]
