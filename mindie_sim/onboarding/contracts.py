"""Immutable contracts, ABI/cliff planning, Lean/receipt checks, and gate runners.

[Role] Define shared immutable dataclasses, domain constants, ISA-scoped
  ``TilingAxes`` / ``validate_abi`` rules, deterministic 512..400K Anchor +
  Off-Grid Cliff plan synthesis, SHA-256 receipt sealing, formal optimality-certificate
  ``OptimalityCertificate`` non-inversion checks, privacy sanitization, bounded
  atomic JSON I/O, and bounded subprocess repo-gate / tail-diagnosis runners.
[Invariants] Every ``TimeoutBudget`` enforces
  ``outer_timeout_s >= sum(inner_budgets) + cleanup_grace_s``. All emitted
  payloads reject raw IPv4 quads, private host paths, and usernames. Unified
  physical tolerance is ``|pred - obs| <= max(rtol * obs, atol_ns)`` with
  ``0 < rtol <= 10%`` and ``0 <= atol_ns <= 2000``. ``TimeoutError`` always
  propagates before ``OSError``.
[Timeouts & Locks] Bounded 50 MiB JSON reads, atomic ``os.replace`` writes, and
  120s bounded repo-gate subprocesses; zero locks or device contexts.
[Side-Effect Policy] Pure helpers except explicit ``write_atomic_json`` and
  caller-requested diagnosis output writes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Literal

from mindie_sim.observations.multilevel import (
    SHA256_HEX_LENGTH, FrozenFitManifest, FrozenFitParameter, normalize_lineage_token,
    sanitize_capture_uri, validate_provenance_text,
)
from mindie_sim.optimization.dual_bound import (
    DagEdge, EventDagSpec, FarkasCertificate, OptimalityCertificate, ResourceChoiceProgram, check_optimality_certificate,
)
from mindie_sim.schema.common import ModelingError

SKU_ONBOARDING_RECEIPT_SCHEMA = "mindie-sim.sku-onboarding-receipt/1"
OPERATOR_ONBOARDING_RECEIPT_SCHEMA = "mindie-sim.operator-onboarding-receipt/1"
CLIFF_PLAN_SCHEMA = "mindie-sim.cliff-bench-plan/2"
SOC_PROBE_SCHEMA = "mindie-sim.soc-probe/1"
TRAIN_OBSERVATION_MANIFEST_SCHEMA = "mindie-sim.train-observation-manifest/1"
VALID_CLOCK_OPERATORS: tuple[str, ...] = ("busy_union", "elapsed_span")
DEFAULT_CLOCK_OPERATOR = "elapsed_span"
STANDARD_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {"L1": "device_kernel", "L2": "fx_region", "L3": "model_step"}
FORBIDDEN_CALLER_PREDICTION_KEYS: frozenset[str] = frozenset({"predicted_ns", "predicted_ms", "predicted_us", "predicted_s", "predicted_duration"})

ARCH_PRESET_22 = "npu_arch_2201"
ARCH_PRESET_35 = "npu_arch_3510"
DEFAULT_HELDOUT_APE_THRESHOLD_PCT = 10.0
DEFAULT_PHYSICAL_ATOL_NS = 2000
ANCHOR_CONTEXT_GRID: tuple[int, ...] = (512, 2048, 8192, 32768, 131072, 262144, 409600)
OFF_GRID_CLIFF_HELDOUT_GRID: tuple[int, ...] = (768, 2049, 4097, 24576, 98304, 200000, 350000, 400000)
CONTEXT_MIN_TOKENS = 512
CONTEXT_MAX_TOKENS = 409600
DEFAULT_SPARSE_TOPK = 2048
LIGHTNING_TOPK_ABOVE_THRESHOLD = 2048
LIGHTNING_TOPK_DIVISOR = 1024
LIGHTNING_TOPK_MAX_VALUE = 8192
QLI_SMALL_M_TOPK_THRESHOLD = 64
DEFAULT_TILE_DIM = 128
COLD_RING_L2_MULTIPLIER = 3
PERCENT_SCALE = 100.0
NS_PER_MS = 1_000_000
NS_PER_US = 1000
MIB_BYTES = 1024 * 1024
GIB_BYTES = 1024 * 1024 * 1024
MAX_READ_JSON_BYTES = 50 * MIB_BYTES
DISPLAY_ROUND_DIGITS = 4
HEX_COMMIT_LEN = 40
SHORT_DIGEST_LEN = 16
MAX_CALIBRATED_CV_PCT = 5.0
DEFAULT_SSH_CONNECT_TIMEOUT_S = 8
DEFAULT_PROBE_TIMEOUT_S = 30
DEFAULT_GATE_SUBPROCESS_TIMEOUT_S = 120
DEFAULT_L0_BUDGET_S = 360
DEFAULT_FABRIC_BUDGET_S = 240
DEFAULT_CLEANUP_GRACE_S = 60
DEFAULT_OUTER_TIMEOUT_S = DEFAULT_L0_BUDGET_S + DEFAULT_FABRIC_BUDGET_S + DEFAULT_CLEANUP_GRACE_S
DEFAULT_TILING_BATCH_AXIS: tuple[int, ...] = (1, 16, 64, 128)
DEFAULT_TILING_QUERY_AXIS: tuple[int, ...] = (1, 2, 4, 8, 2048, 8192)
_CROSS_DECODE_PAIRS: tuple[tuple[int, int], ...] = ((16, 4097), (64, 24576), (128, 98304))
_CROSS_MTP_PAIRS: tuple[tuple[int, int], ...] = ((2, 2049), (4, 24576), (8, 98304))
_CROSS_CHUNK_PAIRS: tuple[tuple[int, int], ...] = ((2048, 98304), (8192, 200000))
_CROSS_PARALLEL_SPECS: tuple[tuple[str, int], ...] = (("tp2_pcp2_dcp2", 200000), ("tp2_pcp2_dcp4", 400000))
_SPARSE_INDEXER_FAMILIES = frozenset({"sparse_indexer", "lightning_indexer", "indexer"})
_CORE_COORD_KEYS: tuple[str, ...] = ("context_len", "tokens", "batch_size", "query_tokens", "topk", "batch_x_kv_heads")
_IPV4_QUAD_RE = re.compile(r"(?<!\d)\d{1,3}(?:[._-]\d{1,3}){3}(?!\d)")
_SAFE_SOC_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
_EMAIL_OR_USER_HOST_RE = re.compile(
    r"(?:^|[\s\"'(<:,=])[A-Za-z_][A-Za-z0-9._%+-]*@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?=$|[\s\"')>,;])"
    r"|[A-Za-z_][A-Za-z0-9._-]*@(?:\d{1,3}\.){3}\d{1,3}"
)
_HOME_ENV_RE = re.compile(r"\$\{?HOME\}?")
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_VIOLATION_COUNT_RE = re.compile(r"(?:violations|findings):\s*(\d+)")
_OPAQUE_HEX_LEN = 12
_OPAQUE_ALIAS_RE = re.compile(r"^ep-[0-9a-f]{12}$")
_OPAQUE_CAMPAIGN_RE = re.compile(r"^camp-[0-9a-f]{12}$")
_OPAQUE_RUN_RE = re.compile(r"^run-[0-9a-f]{12}$")
_OPAQUE_LEASE_RE = re.compile(r"^lease-[0-9a-f]{16}$")
_SAFE_HEX_CAPTURE_NAME_RE = re.compile(r"^[0-9a-f]{64}(?:\.jsonl?)?$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_DIAGNOSE_ESCALATED_EXIT = 1
_PREVIEW_LINES = 8
_PREVIEW_CHAR_LIMIT = 320
_TIMING_TRIPLE_LEN = 3
_TIMING_QUAD_LEN = 4
_LINEAGE_EDGE_PAIR_LEN = 2
_MAX_MISSING_CASES_IN_MSG = 8
_SUBPROCESS_NO_WINDOW = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def subprocess_window_kwargs() -> dict[str, Any]:
    return {"creationflags": _SUBPROCESS_NO_WINDOW} if _SUBPROCESS_NO_WINDOW else {}


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _opaque_tag(raw: str, prefix: str, pattern: re.Pattern[str], fallback: str) -> str:
    text = (raw or "").strip()
    if not text:
        return fallback
    return text if pattern.fullmatch(text) else f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:_OPAQUE_HEX_LEN]}"


def sanitize_public_alias(raw: str, fallback: str = "local") -> str:
    return _opaque_tag(raw, "ep", _OPAQUE_ALIAS_RE, fallback)


def sanitize_public_campaign_id(raw: str, fallback: str = "camp-default") -> str:
    return _opaque_tag(raw, "camp", _OPAQUE_CAMPAIGN_RE, fallback)


def sanitize_public_run_id(raw: str, fallback: str = "run-000000000000") -> str:
    return _opaque_tag(raw, "run", _OPAQUE_RUN_RE, fallback)


def sanitize_public_warning(raw: Any, fallback: str = "receipt_warning") -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    head = re.sub(r"[^a-z0-9_]+", "_", text.split(":", 1)[0].strip().lower()).strip("_")
    code = head if head in {"receipt_warning", "receipt_unconfirmed", "host_queue_lease_acquire_unconfirmed", "host_queue_lease_release_unconfirmed", "release_delayed", "acquire_delayed"} else fallback
    return f"{code}:ref={hashlib.sha256(text.encode('utf-8')).hexdigest()[:_OPAQUE_HEX_LEN]}"


def sanitize_public_capture_uri(raw: str, *, raw_sha256: str = "", default_prefix: str = "capture://onboarding") -> str:
    text = str(raw or "").strip()
    if text:
        clean = sanitize_capture_uri(text)
        if clean.startswith("capture://"):
            parts = [p for p in clean[len("capture://"):].strip("/").split("/") if p]
            if len(parts) == 2 and parts[0] in {"train", "heldout", "onboarding", "soc-probe", "l0"} and _SAFE_HEX_CAPTURE_NAME_RE.fullmatch(parts[1]):
                return f"capture://{parts[0]}/{parts[1]}"
    dig = str(raw_sha256 or "").strip().lower()
    tail = f"{dig}.json" if len(dig) == SHA256_HEX_LENGTH else f"{hashlib.sha256((text or 'cap').encode('utf-8')).hexdigest()[:SHORT_DIGEST_LEN]}.json"
    return f"{default_prefix.rstrip('/')}/{tail}"


def build_capture_uri(namespace: str, payload: bytes) -> str:
    ns = re.sub(r"[^a-z0-9._/-]+", "-", (namespace or "onboarding").strip().lower()).strip("-/") or "onboarding"
    return sanitize_capture_uri(f"capture://{ns}/{sha256_bytes(payload)}")


def sanitize_lease_for_receipt(lease: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(lease, Mapping):
        return {}
    out: dict[str, Any] = {}
    if lease.get("lease_mode") is not None:
        out["lease_mode"] = str(lease["lease_mode"])
    raw_id = str(lease.get("lease_id") or lease.get("lease_digest") or "").strip()
    if raw_id:
        out["lease_digest"] = raw_id if _OPAQUE_LEASE_RE.fullmatch(raw_id) else f"lease-{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:SHORT_DIGEST_LEN]}"
    for src_k, dst_k in (("fence_token", "fence_token"), ("ttl_seconds", "ttl_seconds")):
        if lease.get(src_k) is not None and not isinstance(lease[src_k], bool):
            out[dst_k] = int(lease[src_k])
    if isinstance(lease.get("devices"), (list, tuple)):
        out["device_count"] = len(lease["devices"])
    elif lease.get("device_count") is not None and not isinstance(lease["device_count"], bool):
        out["device_count"] = int(lease["device_count"])
    if bool(lease.get("receipt_unconfirmed")):
        out["receipt_unconfirmed"] = True
    if lease.get("receipt_warning"):
        out["receipt_warning"] = sanitize_public_warning(lease["receipt_warning"])
    return out


def collect_receipt_warnings(*records: Mapping[str, Any] | None) -> tuple[bool, str]:
    unconfirmed, warnings = False, []
    for rec in records:
        if isinstance(rec, Mapping):
            unconfirmed = unconfirmed or bool(rec.get("receipt_unconfirmed"))
            msg = sanitize_public_warning(rec.get("receipt_warning"))
            if msg and msg not in warnings:
                warnings.append(msg)
    return unconfirmed, "; ".join(warnings)


def assert_sanitized_payload(payload: Any, *, field_name: str = "payload") -> None:
    if isinstance(payload, str):
        validate_provenance_text(payload, field_name=field_name)
        for rx, code in ((_IPV4_QUAD_RE, "private_ipv4_rejected"), (_EMAIL_OR_USER_HOST_RE, "private_user_host_rejected"), (_HOME_ENV_RE, "private_home_env_rejected")):
            if rx.search(payload):
                raise ModelingError(f"{code}:{field_name}")
    elif isinstance(payload, Mapping):
        for k, v in payload.items():
            assert_sanitized_payload(v, field_name=f"{field_name}.{k}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_sanitized_payload(item, field_name=f"{field_name}[{i}]")


def read_bounded_json(path: str | Path) -> Any:
    target = Path(path)
    try:
        if target.stat().st_size > MAX_READ_JSON_BYTES:
            raise ModelingError(f"onboarding_json_exceeds_50_mib:{target.name}")
        raw = target.read_bytes()
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_json_read_failed:{target.name}:{exc}") from exc
    if len(raw) > MAX_READ_JSON_BYTES:
        raise ModelingError(f"onboarding_json_exceeds_50_mib:{target.name}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ModelingError(f"onboarding_json_invalid:{target.name}:{exc}") from exc


def write_atomic_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        return target
    except TimeoutError:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    except OSError as exc:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise ModelingError(f"onboarding_json_write_failed:{target.name}:{exc}") from exc


def write_atomic_json(path: str | Path, payload: Any) -> Path:
    return write_atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


@dataclass(frozen=True, slots=True)
class TimeoutBudget:
    """Bounded execution budget enforcing ``outer >= sum(inner) + cleanup_grace``."""

    inner_budgets_s: tuple[tuple[str, int], ...]
    cleanup_grace_s: int = DEFAULT_CLEANUP_GRACE_S
    outer_timeout_s: int = DEFAULT_OUTER_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.inner_budgets_s:
            raise ModelingError("timeout_budget_empty_inner")
        for label, budget in self.inner_budgets_s:
            if not label or type(budget) is not int or budget < 1:
                raise ModelingError(f"timeout_budget_inner_invalid:{label}:{budget}")
        if type(self.cleanup_grace_s) is not int or self.cleanup_grace_s < 1:
            raise ModelingError(f"timeout_budget_grace_invalid:{self.cleanup_grace_s}")
        min_outer = sum(v for _, v in self.inner_budgets_s) + self.cleanup_grace_s
        if type(self.outer_timeout_s) is not int or self.outer_timeout_s < min_outer:
            raise ModelingError(f"timeout_budget_outer_too_small:outer={self.outer_timeout_s}:min_required={min_outer}")

    @property
    def total_inner_s(self) -> int:
        return sum(v for _, v in self.inner_budgets_s)

    def to_dict(self) -> dict[str, Any]:
        return {"inner_budgets_s": {k: v for k, v in self.inner_budgets_s}, "cleanup_grace_s": self.cleanup_grace_s, "outer_timeout_s": self.outer_timeout_s}


@dataclass(frozen=True, slots=True)
class SiliconProfileConflict:
    field: str
    silicon_value: Any
    profile_value: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SocProbeRecord:
    soc_name: str
    arch_preset: str
    architecture: str
    aic_count: int
    aiv_count: int
    hbm_bytes: int
    l2_bytes: int
    ub_bytes: int
    dies_per_package: int
    ranks_per_node: int
    aicore_freq_mhz: int = 0
    l1_bytes: int = 0
    l0a_bytes: int = 0
    l0b_bytes: int = 0
    l0c_bytes: int = 0
    endpoint_alias: str = "local"
    cann_version: str = "unmeasured"
    platform_ini_sha256: str = ""
    capture_uri: str = ""
    fabric_channels: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not _SAFE_SOC_NAME_RE.fullmatch(self.soc_name):
            raise ModelingError(f"soc_name_invalid:{self.soc_name!r}")
        for fname, val in (("aic_count", self.aic_count), ("aiv_count", self.aiv_count), ("dies_per_package", self.dies_per_package), ("ranks_per_node", self.ranks_per_node)):
            if type(val) is not int or val < 0:
                raise ModelingError(f"soc_probe_field_invalid:{fname}:{val}")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schema": SOC_PROBE_SCHEMA, "fabric_channels": dict(self.fabric_channels)}


@dataclass(frozen=True, slots=True)
class AbiDivisibilityRule:
    axis: str
    above_threshold: int
    divisor: int
    max_value: int | None = None
    arch_presets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.axis or type(self.divisor) is not int or self.divisor < 1 or type(self.above_threshold) is not int or self.above_threshold < 0:
            raise ModelingError(f"abi_rule_invalid:{self.axis}:{self.above_threshold}:{self.divisor}")
        if self.max_value is not None and (type(self.max_value) is not int or self.max_value < 1):
            raise ModelingError(f"abi_rule_max_invalid:{self.axis}:{self.max_value}")

    def applies_to(self, arch_preset: str) -> bool:
        return not self.arch_presets or (bool(arch_preset) and arch_preset in self.arch_presets)

    def check(self, value: int) -> tuple[bool, str]:
        if type(value) is not int or value < 1:
            return False, f"abi_nonpositive_value:{self.axis}={value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"abi_max_violation:{self.axis}={value}:max={self.max_value}"
        if value > self.above_threshold and self.divisor > 1 and (value % self.divisor) != 0:
            return False, f"abi_divisibility_violation:{self.axis}={value}:above={self.above_threshold}:divisor={self.divisor}"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "above_threshold": self.above_threshold, "divisor": self.divisor, "max_value": self.max_value, "arch_presets": list(self.arch_presets)}


def default_topk_abi_rule(
    *, above_threshold: int = LIGHTNING_TOPK_ABOVE_THRESHOLD, divisor: int = LIGHTNING_TOPK_DIVISOR,
    max_value: int = LIGHTNING_TOPK_MAX_VALUE, arch_presets: tuple[str, ...] = (ARCH_PRESET_22,),
) -> AbiDivisibilityRule:
    return AbiDivisibilityRule("topk", above_threshold, divisor, max_value, arch_presets)


@dataclass(frozen=True, slots=True)
class TilingAxes:
    tile_dims: tuple[int, ...] = (DEFAULT_TILE_DIM,)
    core_product_axes: tuple[str, ...] = ("tokens", "batch_x_kv_heads")
    phase_thresholds: tuple[tuple[str, int], ...] = ()
    abi_divisibility: tuple[AbiDivisibilityRule, ...] = ()
    batch_axis: tuple[int, ...] = DEFAULT_TILING_BATCH_AXIS
    query_tokens_axis: tuple[int, ...] = DEFAULT_TILING_QUERY_AXIS
    enforce_topk_le_context_archs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_dims": list(self.tile_dims), "core_product_axes": list(self.core_product_axes),
            "phase_thresholds": dict(self.phase_thresholds), "abi_divisibility": [r.to_dict() for r in self.abi_divisibility],
            "batch_axis": list(self.batch_axis), "query_tokens_axis": list(self.query_tokens_axis),
            "enforce_topk_le_context_archs": list(self.enforce_topk_le_context_archs),
        }


def family_default_tiling_axes(
    family: str, *, tile_dim: int, arch_preset: str = "", topk_threshold: int | None = None,
) -> TilingAxes:
    clean = str(family or "").strip()
    for mod_path in (f"mindie_sim.prediction.operators.{clean}.common", f"mindie_sim.prediction.operators.{clean}"):
        if not clean:
            break
        try:
            mod = importlib.import_module(mod_path)
            decl = getattr(mod, "TILING_AXES", None)
            if callable(decl):
                res = decl(arch_preset)
                if isinstance(res, TilingAxes):
                    return res
            elif isinstance(decl, TilingAxes):
                return decl
        except Exception:
            continue
    if clean in _SPARSE_INDEXER_FAMILIES or topk_threshold is not None:
        tk = int(topk_threshold) if topk_threshold is not None else LIGHTNING_TOPK_ABOVE_THRESHOLD
        bounds = default_topk_abi_rule(divisor=1, arch_presets=())
        dav22 = default_topk_abi_rule(divisor=LIGHTNING_TOPK_DIVISOR, arch_presets=(ARCH_PRESET_22,))
        thr: tuple[tuple[str, int], ...] = (
            (("topk", tk), ("topk", QLI_SMALL_M_TOPK_THRESHOLD))
            if (clean in _SPARSE_INDEXER_FAMILIES and arch_preset == ARCH_PRESET_35 and tk != QLI_SMALL_M_TOPK_THRESHOLD)
            else (("topk", tk),)
        )
        return TilingAxes(tile_dims=(int(tile_dim),), phase_thresholds=thr, abi_divisibility=(bounds, dav22))
    return TilingAxes(tile_dims=(int(tile_dim),))


def validate_abi(point: Mapping[str, Any], axes: TilingAxes, *, arch_preset: str = "") -> tuple[bool, str]:
    axis_name = str(point.get("axis") or "").strip()
    resolved: dict[str, int] = {}
    if axis_name and point.get("value") is not None:
        if type(point["value"]) is not int:
            return False, f"abi_type_violation:{axis_name}={point['value']!r}"
        resolved[axis_name] = point["value"]
    for key in dict.fromkeys((*_CORE_COORD_KEYS, *(r.axis for r in axes.abi_divisibility))):
        if key not in point or point[key] is None:
            continue
        if type(point[key]) is not int:
            return False, f"abi_type_violation:{key}={point[key]!r}"
        resolved[key] = point[key]
    for rule in axes.abi_divisibility:
        if rule.applies_to(arch_preset) and rule.axis in resolved:
            ok, reason = rule.check(resolved[rule.axis])
            if not ok:
                return False, reason
    if arch_preset and arch_preset in axes.enforce_topk_le_context_archs and "topk" in resolved and "context_len" in resolved and resolved["topk"] > resolved["context_len"]:
        return False, f"abi_topk_exceeds_context:topk={resolved['topk']}>context={resolved['context_len']}"
    return True, ""


@dataclass(frozen=True, slots=True)
class CliffPointSpec:
    point_id: str
    split: Literal["train", "heldout", "abi_rejection"]
    axis: str
    value: int
    regime_tag: str
    rationale: str
    batch_size: int = 1
    query_tokens: int = 1
    topk: int | None = None
    parallel_topology: str = "tp1"
    clock_operator: str = DEFAULT_CLOCK_OPERATOR
    abi_expected_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperatorCliffParameters:
    operator_id: str
    family: str
    tile_dim: int
    core_count: int
    l2_bytes: int
    bytes_per_token: int = 0
    topk_threshold: int | None = None
    context_min: int = CONTEXT_MIN_TOKENS
    context_max: int = CONTEXT_MAX_TOKENS
    tiling_axes: TilingAxes | None = None


@dataclass(frozen=True, slots=True)
class CliffBenchPlan:
    target_id: str
    sku: str
    train_lineage_root: str
    heldout_lineage_root: str
    train_points: tuple[CliffPointSpec, ...]
    heldout_points: tuple[CliffPointSpec, ...]
    plan_sha256: str
    abi_rejection_points: tuple[CliffPointSpec, ...] = ()
    arch_preset: str = ""
    tiling_axes: TilingAxes | None = None
    family: str = ""
    campaign_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLIFF_PLAN_SCHEMA, "target_id": self.target_id, "sku": self.sku,
            "family": self.family, "campaign_id": self.campaign_id, "arch_preset": self.arch_preset,
            "tiling_axes": self.tiling_axes.to_dict() if self.tiling_axes is not None else None,
            "train_lineage_root": self.train_lineage_root, "heldout_lineage_root": self.heldout_lineage_root,
            "train_points": [p.to_dict() for p in self.train_points],
            "heldout_points": [p.to_dict() for p in self.heldout_points],
            "abi_rejection_points": [p.to_dict() for p in self.abi_rejection_points],
            "plan_sha256": self.plan_sha256,
        }


def validate_parallel_topology(topo: str) -> int:
    text = str(topo or "tp1").strip().lower()
    tokens = _TOPO_TOKEN_RE.findall(text)
    if not tokens or "_".join(f"{k}{v}" for k, v in tokens) != text:
        raise ModelingError(f"parallel_topology_invalid:{topo}")
    axes = {k: int(v) for k, v in tokens}
    if any(v < 1 for v in axes.values()):
        raise ModelingError(f"parallel_topology_nonpositive:{topo}")
    tp, pcp, dcp = axes.get("tp", 1), axes.get("pcp", 1), axes.get("dcp", 1)
    if dcp > 1 and pcp > 1 and (dcp % pcp != 0):
        raise ModelingError(f"dcp_pcp_domain_invalid:dcp={dcp}:pcp={pcp}:{topo}")
    if dcp > 1 and dcp > tp * pcp:
        raise ModelingError(f"parallel_topology_invalid:{topo}")
    return max(1, tp * max(1, axes.get("dp", 1)) * max(1, axes.get("pp", 1)) * max(1, pcp))


def required_ranks_for_plan(plan_doc: Mapping[str, Any]) -> int:
    max_r = 1
    for split_key in ("train_points", "heldout_points", "abi_rejection_points"):
        for pt in plan_doc.get(split_key) or ():
            if isinstance(pt, Mapping):
                max_r = max(max_r, validate_parallel_topology(str(pt.get("parallel_topology") or "tp1")))
    return max_r


def _resolve_plan_arch_preset(sku: str, arch_preset: str, profile_override: Mapping[str, Any] | None) -> str:
    if arch_preset.strip():
        return arch_preset.strip()
    from .whitebox_runner import resolve_sku_profile

    prof = resolve_sku_profile(sku, profile_override)
    if prof is None:
        raise ModelingError(f"unknown_sku_profile:{sku}")
    dev = prof.get("device") if isinstance(prof.get("device"), Mapping) else {}
    ca = dev.get("compute_arch") if isinstance(dev.get("compute_arch"), Mapping) else {}
    preset = str(ca.get("preset") or "").strip()
    if not preset:
        raise ModelingError(f"sku_profile_missing_arch_preset:{sku}")
    return preset


def _plan_digest_payload(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {k: doc.get(k) for k in ("target_id", "sku", "family", "arch_preset", "train_lineage_root", "heldout_lineage_root", "train_points", "heldout_points", "abi_rejection_points")}


def verify_cliff_plan_integrity(plan_doc: Mapping[str, Any]) -> str:
    expected = str(plan_doc.get("plan_sha256") or "").strip().lower()
    actual = sha256_bytes(canonical_json_bytes(_plan_digest_payload(plan_doc)))
    if not expected or expected != actual:
        raise ModelingError(f"cliff_plan_sha256_tampered:expected={expected[:SHORT_DIGEST_LEN]}:actual={actual[:SHORT_DIGEST_LEN]}")
    return actual


def synthesize_operator_cliff_plan(
    params: OperatorCliffParameters, *, sku: str, campaign_id: str = "onboard",
    arch_preset: str = "", profile_override: Mapping[str, Any] | None = None,
) -> CliffBenchPlan:
    resolved_arch = _resolve_plan_arch_preset(sku, arch_preset, profile_override)
    axes = params.tiling_axes or family_default_tiling_axes(params.family, tile_dim=int(params.tile_dim), arch_preset=resolved_arch, topk_threshold=params.topk_threshold)
    clean_camp = sanitize_public_campaign_id(campaign_id, "camp-onboard")
    train_root = f"{clean_camp}-{sha256_bytes(f'train:{sku}:{params.operator_id}:{clean_camp}'.encode('utf-8'))[:SHORT_DIGEST_LEN]}"
    held_root = f"{clean_camp}-{sha256_bytes(f'heldout:{sku}:{params.operator_id}:{clean_camp}'.encode('utf-8'))[:SHORT_DIGEST_LEN]}"
    train_pts: list[CliffPointSpec] = []
    held_pts: list[CliffPointSpec] = []
    abi_rejs: list[CliffPointSpec] = []
    seen_ho: set[tuple[Any, ...]] = set()
    seen_rej: set[tuple[Any, ...]] = set()

    def _add(pid: str, axis: str, val: int, tag: str, why: str, *, b: int = 1, q: int = 1, tk: int | None = None, topo: str = "tp1") -> None:
        validate_parallel_topology(topo)
        ok, err = validate_abi({"axis": axis, "value": val, "batch_size": b, "query_tokens": q, "topk": tk}, axes, arch_preset=resolved_arch)
        key = (axis, val, b, q, tk, topo)
        if not ok and key not in seen_rej:
            seen_rej.add(key)
            abi_rejs.append(CliffPointSpec(f"abi_reject_{pid}", "abi_rejection", axis, val, f"{tag}_abi_reject", why, b, q, tk, topo, DEFAULT_CLOCK_OPERATOR, err))
        elif ok and key not in seen_ho:
            seen_ho.add(key)
            held_pts.append(CliffPointSpec(pid, "heldout", axis, val, tag, why, b, q, tk, topo))

    def _in_ctx(ctx_val: int) -> bool:
        return params.context_min <= ctx_val <= params.context_max

    train_pts.extend(CliffPointSpec(f"anchor_ctx_{c}", "train", "context_len", c, "anchor_grid", "Anchor context grid") for c in ANCHOR_CONTEXT_GRID if _in_ctx(c))
    for c in OFF_GRID_CLIFF_HELDOUT_GRID:
        if _in_ctx(c):
            _add(f"offgrid_ctx_{c}", "context_len", c, "off_grid_heldout", "Off-grid context cliff")
    for b, c in _CROSS_DECODE_PAIRS:
        if _in_ctx(c):
            _add(f"cross_b{b}_ctx_{c}", "context_len", c, "high_batch_decode_2d", "High-batch decode 2D", b=b)
    for pairs, prefix, tag, why in ((_CROSS_MTP_PAIRS, "cross_mtp_q", "mtp_decode_2d", "MTP decode 2D"), (_CROSS_CHUNK_PAIRS, "cross_chunk_q", "chunked_prefill_2d", "Chunked prefill 2D")):
        for q, c in pairs:
            if _in_ctx(c):
                _add(f"{prefix}{q}_ctx_{c}", "context_len", c, tag, why, q=q)
    for topo, c in _CROSS_PARALLEL_SPECS:
        if _in_ctx(c):
            _add(f"cross_{topo}_ctx_{c}", "context_len", c, "parallel_split_2d", "Multi-rank TP/PCP/DCP", topo=topo)
    for td in axes.tile_dims:
        for delta, suffix in ((-1, "minus1"), (0, "exact"), (1, "plus1")):
            if td + delta >= 1:
                _add(f"tile_{td}_{suffix}_{td + delta}", "tokens", td + delta, "tile_boundary", f"Tile boundary {td}{delta:+d}")
        prod = max(1, int(params.core_count)) * td
        for c_axis in axes.core_product_axes:
            for delta, suffix in ((-1, "minus1"), (1, "plus1")):
                if prod + delta >= 1:
                    _add(f"cores_wave_{c_axis}_{td}_{suffix}_{prod + delta}", c_axis, prod + delta, "core_wave_boundary", f"Core wave {prod}{delta:+d}")
    for p_axis, thr in axes.phase_thresholds:
        for delta, suffix in ((-1, "minus1"), (1, "plus1")):
            if (v := thr + delta) >= 1:
                if p_axis == "topk":
                    _add(f"topk_attr_{v}", "context_len", max(v, CONTEXT_MIN_TOKENS), "topk_phase_cliff", f"Top-K {thr}{delta:+d}", tk=v)
                else:
                    _add(f"phase_{p_axis}_{suffix}_{v}", p_axis, v, "phase_threshold", f"Phase threshold {thr}{delta:+d}")
    for rule in axes.abi_divisibility:
        if rule.applies_to(resolved_arch) and rule.max_value is not None:
            over = rule.max_value + 1
            is_tk = rule.axis == "topk"
            _add(f"{rule.axis}_over_max_{over}", "context_len" if is_tk else rule.axis, max(over, CONTEXT_MIN_TOKENS) if is_tk else over, "abi_max_boundary", f"ABI max+1 ({over})", tk=over if is_tk else None)

    draft = {
        "target_id": params.operator_id, "sku": sku, "family": params.family, "arch_preset": resolved_arch,
        "train_lineage_root": train_root, "heldout_lineage_root": held_root,
        "train_points": [p.to_dict() for p in train_pts],
        "heldout_points": [p.to_dict() for p in held_pts],
        "abi_rejection_points": [p.to_dict() for p in abi_rejs],
    }
    return CliffBenchPlan(
        target_id=params.operator_id, sku=sku, train_lineage_root=train_root, heldout_lineage_root=held_root,
        train_points=tuple(train_pts), heldout_points=tuple(held_pts), abi_rejection_points=tuple(abi_rejs),
        plan_sha256=sha256_bytes(canonical_json_bytes(draft)), arch_preset=resolved_arch,
        tiling_axes=axes, family=params.family, campaign_id=clean_camp,
    )


def synthesize_sku_512_400k_plan(
    *, sku: str, core_count: int, l2_bytes: int, tile_dim: int = DEFAULT_TILE_DIM,
    campaign_id: str = "sku-512-400k", arch_preset: str = "", profile_override: Mapping[str, Any] | None = None,
) -> CliffBenchPlan:
    resolved_arch = _resolve_plan_arch_preset(sku, arch_preset, profile_override)
    axes = family_default_tiling_axes("sparse_indexer", tile_dim=tile_dim, arch_preset=resolved_arch, topk_threshold=DEFAULT_SPARSE_TOPK)
    return synthesize_operator_cliff_plan(
        OperatorCliffParameters(f"sku:{sku}:512_400k", "sku_suite", tile_dim, core_count, l2_bytes, topk_threshold=DEFAULT_SPARSE_TOPK, tiling_axes=axes),
        sku=sku, campaign_id=campaign_id, arch_preset=resolved_arch, profile_override=profile_override,
    )


def evaluate_abi_rejections(
    planned_rejections: Sequence[Mapping[str, Any]], observed_rejections: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if not planned_rejections:
        return {"ok": True, "required": 0, "verified": 0, "missing": [], "mismatches": []}
    by_id = {str(r.get("point_id") or r.get("case_id") or ""): r for r in (observed_rejections or ()) if isinstance(r, Mapping)}
    missing, mismatches, verified = [], [], 0
    for pt in planned_rejections:
        pid = str(pt.get("point_id") or "")
        exp = str(pt.get("abi_expected_error") or "").strip()
        obs = by_id.get(pid)
        if obs is None:
            missing.append(pid)
            continue
        if not bool(obs.get("rejected")):
            mismatches.append(f"{pid}:not_rejected")
            continue
        err = str(obs.get("abi_error") or "").strip()
        if exp and exp not in err:
            mismatches.append(f"{pid}:expected={exp}:got={err}")
            continue
        verified += 1
    return {"ok": not missing and not mismatches, "required": len(planned_rejections), "verified": verified, "missing": missing, "mismatches": mismatches}


@dataclass(frozen=True, slots=True)
class GateCheckOutcome:
    name: str
    passed: bool
    status: str
    detail: str
    metrics: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "status": self.status, "detail": self.detail, "metrics": dict(self.metrics)}


@dataclass(frozen=True, slots=True)
class LeanBoundCheckRow:
    case_id: str
    l_star_lean_ns: int
    t_sim_ns: int
    t_silicon_p99_ns: int
    certificate_checked: bool
    certificate_valid: bool
    sim_inverted: bool
    silicon_inverted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OnboardingReceipt:
    schema: str
    target_kind: Literal["sku", "operator"]
    target_id: str
    sku: str
    status: str
    ok: bool
    created_utc: str
    gates: tuple[GateCheckOutcome, ...]
    artifacts: tuple[tuple[str, str], ...]
    summary: tuple[tuple[str, Any], ...]
    receipt_sha256: str = ""
    receipt_unconfirmed: bool = False
    receipt_warning: str = ""

    def to_unsigned_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": self.schema, "target_kind": self.target_kind, "target_id": self.target_id,
            "sku": self.sku, "status": self.status, "ok": self.ok, "created_utc": self.created_utc,
            "gates": [g.to_dict() for g in self.gates], "artifacts": dict(self.artifacts), "summary": dict(self.summary),
        }
        if self.receipt_unconfirmed or self.receipt_warning:
            out["receipt_unconfirmed"], out["receipt_warning"] = self.receipt_unconfirmed, self.receipt_warning
        return out

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_unsigned_dict(), "receipt_sha256": self.receipt_sha256}


def compute_receipt_sha256(unsigned_receipt: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({k: v for k, v in unsigned_receipt.items() if k != "receipt_sha256"}))


def verify_receipt_sha256(receipt: Mapping[str, Any]) -> bool:
    claimed = str(receipt.get("receipt_sha256") or "").strip().lower()
    return len(claimed) == SHA256_HEX_LENGTH and claimed == compute_receipt_sha256(receipt)


def seal_onboarding_receipt(
    *, schema: str, target_kind: Literal["sku", "operator"], target_id: str, sku: str,
    status: str, ok: bool, gates: Sequence[GateCheckOutcome], artifacts: Mapping[str, str],
    summary: Mapping[str, Any], receipt_unconfirmed: bool = False, receipt_warning: str = "",
) -> OnboardingReceipt:
    draft = OnboardingReceipt(
        schema, target_kind, target_id, sku, status, ok, utc_now_iso(), tuple(gates),
        tuple(sorted((str(k), str(v)) for k, v in artifacts.items())),
        tuple(sorted((str(k), v) for k, v in summary.items())),
        receipt_unconfirmed=bool(receipt_unconfirmed),
        receipt_warning=sanitize_public_warning(receipt_warning) if receipt_warning else "",
    )
    unsigned = draft.to_unsigned_dict()
    assert_sanitized_payload(unsigned, field_name="onboarding_receipt")
    return OnboardingReceipt(
        draft.schema, draft.target_kind, draft.target_id, draft.sku, draft.status, draft.ok, draft.created_utc,
        draft.gates, draft.artifacts, draft.summary, compute_receipt_sha256(unsigned),
        draft.receipt_unconfirmed, draft.receipt_warning,
    )


def validate_strict_threshold_pct(threshold_pct: float) -> float:
    if isinstance(threshold_pct, bool) or not isinstance(threshold_pct, (int, float)) or not math.isfinite(float(threshold_pct)):
        raise ModelingError(f"threshold_pct_invalid:{threshold_pct!r}")
    val = float(threshold_pct)
    if val <= 0.0 or val > DEFAULT_HELDOUT_APE_THRESHOLD_PCT:
        raise ModelingError(f"threshold_pct_exceeds_strict_release_cap:{val}:max={DEFAULT_HELDOUT_APE_THRESHOLD_PCT}")
    return val


def validate_strict_atol_ns(atol_ns: int) -> int:
    if isinstance(atol_ns, bool) or not isinstance(atol_ns, int):
        raise ModelingError(f"atol_ns_invalid:{atol_ns!r}")
    if atol_ns < 0 or atol_ns > DEFAULT_PHYSICAL_ATOL_NS:
        raise ModelingError(f"atol_ns_exceeds_strict_release_cap:{atol_ns}:max={DEFAULT_PHYSICAL_ATOL_NS}")
    return atol_ns


def exceeds_physical_tolerance(
    predicted_ns: int, measured_ns: int, *, threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT, atol_ns: int = DEFAULT_PHYSICAL_ATOL_NS,
    allow_signed: bool = False,
) -> bool:
    if not allow_signed and (measured_ns <= 0 or predicted_ns <= 0):
        return True
    return float(abs(int(predicted_ns) - int(measured_ns))) > max((float(threshold_pct) / PERCENT_SCALE) * float(abs(measured_ns)), float(atol_ns))


def program_dag_sha256(program: Mapping[str, Any], dag: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({"program": dict(program), "dag": dict(dag)}))


def parse_lineage_edges(raw_edges: Any) -> tuple[tuple[str, str], ...]:
    """Parse ``(parent, child)`` lineage edge declarations from JSON/mapping sources."""
    edges: list[tuple[str, str]] = []
    for edge in raw_edges or ():
        if isinstance(edge, Mapping):
            parent, child = str(edge.get("parent") or "").strip(), str(edge.get("child") or "").strip()
        elif isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) == _LINEAGE_EDGE_PAIR_LEN:
            parent, child = str(edge[0]).strip(), str(edge[1]).strip()
        else:
            raise ModelingError("frozen_fit_lineage_edge_invalid")
        if not parent or not child:
            raise ModelingError("frozen_fit_lineage_edge_invalid")
        edges.append((parent, child))
    return tuple(edges)


def load_frozen_fit_manifest(
    fits_source: str | Path | Mapping[str, Any] | FrozenFitManifest | None,
    *, default_fit_id: str = "onboarding-fits",
) -> tuple[FrozenFitManifest, str]:
    """Load and canonically digest a ``FrozenFitManifest`` from a path, mapping, or instance."""
    if fits_source is None:
        return FrozenFitManifest(default_fit_id, "mindie-frozen-fits/1", ()), ""
    if isinstance(fits_source, FrozenFitManifest):
        return fits_source, sha256_bytes(canonical_json_bytes(asdict(fits_source)))
    raw = read_bounded_json(fits_source) if isinstance(fits_source, (str, Path)) else fits_source
    if not isinstance(raw, Mapping):
        raise ModelingError("frozen_fit_manifest_not_mapping")
    fit_id = str(raw.get("fit_id") or default_fit_id).strip()
    schema_ver = str(raw.get("schema_version") or "mindie-frozen-fits/1").strip()
    params: list[FrozenFitParameter] = []
    for item in raw.get("parameters") or ():
        if not isinstance(item, Mapping):
            raise ModelingError("frozen_fit_parameter_invalid")
        pid = str(item.get("parameter_id") or "").strip()
        if not pid:
            raise ModelingError("frozen_fit_parameter_id_missing")
        digests = tuple(str(d).strip() for d in (item.get("training_digests") or ()) if str(d).strip())
        params.append(FrozenFitParameter(pid, digests, str(item.get("source_uri") or "").strip()))
    edges = parse_lineage_edges(raw.get("lineage_edges") or ())
    manifest = FrozenFitManifest(fit_id, schema_ver, tuple(params), edges)
    return manifest, sha256_bytes(canonical_json_bytes(asdict(manifest)))


def verify_fits_disjoint_from_residuals(fits: FrozenFitManifest, score_doc: Mapping[str, Any]) -> None:
    """Reject any frozen-fit parameter whose lineage overlaps heldout residual digests or roots."""
    if not fits.parameters:
        return
    heldout_tokens: set[str] = set()
    for row in score_doc.get("residuals") or ():
        if isinstance(row, Mapping) and str(row.get("split") or "") == "heldout":
            for key in ("raw_sha256", "acquisition_root", "run_uid"):
                if (val := str(row.get(key) or "").strip()):
                    heldout_tokens.add(normalize_lineage_token(val))
            for tok in row.get("parent_digests") or ():
                if (val := str(tok or "").strip()):
                    heldout_tokens.add(normalize_lineage_token(val))
    lineage = score_doc.get("lineage")
    extra_edges: tuple[tuple[str, str], ...] = ()
    if isinstance(lineage, Mapping):
        for tok in lineage.get("heldout_roots") or ():
            if (val := str(tok or "").strip()):
                heldout_tokens.add(normalize_lineage_token(val))
        if lineage.get("lineage_edges"):
            extra_edges = parse_lineage_edges(lineage.get("lineage_edges"))
    if not heldout_tokens:
        return
    adj: dict[str, set[str]] = {}
    for parent, child in (*fits.lineage_edges, *extra_edges):
        np_tok, nc_tok = normalize_lineage_token(parent), normalize_lineage_token(child)
        adj.setdefault(np_tok, set()).add(nc_tok)
        adj.setdefault(nc_tok, set()).add(np_tok)
    closure, queue = set(heldout_tokens), list(heldout_tokens)
    while queue:
        cur = queue.pop()
        for nxt in adj.get(cur, ()):
            if nxt not in closure:
                closure.add(nxt)
                queue.append(nxt)
    for param in fits.parameters:
        param_tokens = {normalize_lineage_token(d) for d in param.training_digests if d and d.strip()}
        overlap = sorted(param_tokens & closure)
        if overlap:
            raise ModelingError(f"lineage_fit_heldout_leakage:parameter={param.parameter_id}:roots={','.join(overlap)}")


def verify_capture_file_identity(
    raw_bytes: bytes, *, cid: str, sku: str = "", split: str = "", level: str = "",
    measured_ns: int | None = None, measured_p99_ns: int | None = None, clock_operator: str = "",
) -> str:
    """Verify capture file JSON content matches the residual's case identity and measurements."""
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return f"capture_file_json_invalid:{cid}"
    if not isinstance(doc, Mapping):
        return f"capture_file_json_invalid:{cid}"
    cap_cid = str(doc.get("case_id") or "").strip()
    if not cap_cid or cap_cid != cid:
        return f"capture_file_identity_mismatch:{cid}:case_id={cap_cid or 'missing'}"
    if sku and (cap_sku := str(doc.get("sku") or "").strip()) and cap_sku != sku:
        return f"capture_file_identity_mismatch:{cid}:sku={cap_sku}"
    if split and (cap_split := str(doc.get("split") or "").strip().lower()) and cap_split != split.lower():
        return f"capture_file_identity_mismatch:{cid}:split={cap_split}"
    if level and (cap_lvl := str(doc.get("level") or "").strip().upper()) and cap_lvl != level.upper():
        return f"capture_file_identity_mismatch:{cid}:level={cap_lvl}"
    if measured_ns is not None and doc.get("measured_ns") is not None and int(round(float(doc["measured_ns"]))) != int(measured_ns):
        return f"capture_file_identity_mismatch:{cid}:measured_ns"
    if measured_p99_ns is not None and doc.get("measured_p99_ns") is not None and int(round(float(doc["measured_p99_ns"]))) != int(measured_p99_ns):
        return f"capture_file_identity_mismatch:{cid}:measured_p99_ns"
    if clock_operator and (cap_clk := str(doc.get("clock_operator") or "").strip()) and cap_clk != clock_operator:
        return f"capture_file_identity_mismatch:{cid}:clock_operator={cap_clk}"
    return ""


def _verify_residual_capture_files(
    residuals: Sequence[Mapping[str, Any]], capture_root: str | Path | None, require_capture_files: bool,
) -> tuple[bool, int, str, str]:
    if capture_root is None:
        return (False, 0, "capture_file_missing:capture_root_not_provided", "") if require_capture_files else (True, 0, "optional_not_requested", "")
    root = Path(capture_root)
    if not root.is_dir():
        return False, 0, f"capture_file_missing:{root.name}", ""
    if not residuals:
        return False, 0, "capture_file_missing:no_residuals", ""
    verified_pairs: list[tuple[str, str]] = []
    violations: list[str] = []
    for row in residuals:
        cid = str(row.get("case_id") or "")
        raw_sha = str(row.get("raw_sha256") or "").strip().lower()
        if not _HEX_64_RE.fullmatch(raw_sha):
            violations.append(f"capture_raw_sha256_invalid:{cid}")
            continue
        cand = root / f"{raw_sha}.json"
        if not cand.is_file():
            violations.append(f"capture_file_missing:{cid}:{raw_sha[:SHORT_DIGEST_LEN]}")
            continue
        try:
            if cand.stat().st_size > MAX_READ_JSON_BYTES:
                violations.append(f"capture_file_exceeds_50_mib:{cid}")
                continue
            raw_bytes = cand.read_bytes()
        except TimeoutError:
            raise
        except OSError:
            violations.append(f"capture_file_read_failed:{cid}")
            continue
        if sha256_bytes(raw_bytes).lower() != raw_sha:
            violations.append(f"capture_file_sha256_mismatch:{cid}")
            continue
        id_err = verify_capture_file_identity(
            raw_bytes, cid=cid, sku=str(row.get("sku") or "").strip(),
            split=str(row.get("split") or "").strip(), level=str(row.get("level") or "").strip(),
            measured_ns=int(row["measured_ns"]) if row.get("measured_ns") is not None else None,
            measured_p99_ns=int(row["measured_p99_ns"]) if row.get("measured_p99_ns") is not None else None,
            clock_operator=str(row.get("clock_operator") or "").strip(),
        )
        if id_err:
            violations.append(id_err)
            continue
        verified_pairs.append((cid, raw_sha))
    if violations:
        return False, len(verified_pairs), "; ".join(violations[:_PREVIEW_LINES]), ""
    bundle_sha = sha256_bytes(canonical_json_bytes(sorted(verified_pairs)))
    return True, len(verified_pairs), f"{len(verified_pairs)}/{len(residuals)} capture files verified", bundle_sha


def _strict_int(val: Any, name: str) -> int:
    if type(val) is not int:
        raise ModelingError(f"lean_witness_non_int:{name}:{val!r}")
    return val


def _parse_certificate_witness(row: Mapping[str, Any]) -> tuple[ResourceChoiceProgram, EventDagSpec, OptimalityCertificate] | None:
    p_raw, d_raw, c_raw = row.get("program"), row.get("dag"), row.get("certificate")
    if p_raw is None and d_raw is None and c_raw is None:
        return None
    if not (isinstance(p_raw, Mapping) and isinstance(d_raw, Mapping) and isinstance(c_raw, Mapping)):
        raise ModelingError("lean_certificate_witness_incomplete")
    prog = ResourceChoiceProgram(
        _strict_int(p_raw["num_resources"], "num_resources"),
        tuple(_strict_int(v, "global_work") for v in p_raw["global_work"]),
        tuple(tuple(tuple(_strict_int(x, "site_modes") for x in m) for m in s) for s in p_raw.get("site_modes", ())),
    )
    quanta = tuple(_strict_int(v, "node_quantum_ns") for v in d_raw["node_quantum_ns"])
    edges = tuple(DagEdge(_strict_int(e["src"], "src"), _strict_int(e["dst"], "dst"), _strict_int(e["wire_ns"], "wire_ns")) for e in d_raw.get("edges", ()))
    f_raw = c_raw["farkas"]
    farkas = FarkasCertificate(_strict_int(f_raw["scale_d"], "scale_d"), tuple(_strict_int(v, "lambda_num") for v in f_raw["lambda_num"]), tuple(_strict_int(v, "mu_num") for v in f_raw.get("mu_num", ())), _strict_int(f_raw["lower_bound_ns"], "lower_bound_ns"))
    sink = _strict_int(c_raw["sink_node_id"], "sink_node_id")
    if sink < 0 or sink >= len(quanta):
        raise ModelingError(f"lean_certificate_sink_out_of_range:{sink}")
    cert = OptimalityCertificate(farkas, tuple(_strict_int(v, "path_edges") for v in c_raw.get("path_edges", ())), sink, _strict_int(c_raw["l_star_lean_ns"], "l_star_lean_ns"))
    return prog, EventDagSpec(quanta, edges), cert


def check_lean_non_inversion(
    rows: Sequence[Mapping[str, Any]], *, require_non_empty: bool = True, verify_certificates: bool = True,
    require_captured_p99: bool = True, require_program_dag_join: bool = True,
    expected_case_timings: Mapping[str, tuple[Any, ...]] | None = None,
) -> tuple[bool, tuple[LeanBoundCheckRow, ...], tuple[str, ...]]:
    """Enforce ``0 < L*_Lean <= min(T_sim, T_silicon_p99)`` and valid Lean certificates."""
    if not rows:
        return (False, (), ("lean_bounds_missing",)) if require_non_empty else (True, (), ())
    checked, violations, seen_ids = [], [], set()
    for idx, row in enumerate(rows):
        cid = str(row.get("case_id") or f"case-{idx}")
        dup_case = cid in seen_ids
        seen_ids.add(cid)
        raw_vals = (row.get("l_star_lean_ns"), row.get("t_sim_ns"), row.get("t_silicon_p99_ns"))
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in raw_vals):
            violations.append(f"lean_bound_fields_invalid:{cid}")
            checked.append(LeanBoundCheckRow(cid, -1, 0, 0, False, False, True, True))
            continue
        l_star, t_sim, t_p99 = (int(round(float(v))) for v in raw_vals)
        cert_checked, cert_valid, witness_sha = False, l_star > 0 and t_sim > 0 and t_p99 > 0, ""
        if dup_case:
            cert_valid = False
            violations.append(f"lean_witness_duplicate_case:{cid}")
        try:
            witness = _parse_certificate_witness(row)
        except (ModelingError, KeyError, TypeError, ValueError) as exc:
            witness, cert_valid = None, False
            violations.append(f"lean_certificate_malformed:{cid}:{exc}")
        if witness is not None:
            cert_checked, witness_sha = True, program_dag_sha256(row["program"], row["dag"])
            if not (check_optimality_certificate(*witness) and witness[2].l_star_lean_ns == l_star):
                cert_valid = False
                violations.append(f"lean_certificate_invalid:{cid}")
        elif verify_certificates:
            cert_valid = False
            violations.append(f"lean_certificate_missing:{cid}")
        if expected_case_timings is not None and cid not in expected_case_timings:
            cert_valid = False
            violations.append(f"lean_witness_unexpected_case:{cid}")
        elif expected_case_timings is not None:
            entry = expected_case_timings[cid]
            exp_sim, exp_meas = int(entry[0]), int(entry[1])
            exp_p99 = int(entry[2]) if len(entry) >= _TIMING_TRIPLE_LEN and entry[2] is not None else None
            exp_sha = str(entry[3] or "").strip().lower() if len(entry) >= _TIMING_QUAD_LEN else ""
            exp_sha_ok = bool(_HEX_64_RE.fullmatch(exp_sha))
            wit_sha_ok = bool(_HEX_64_RE.fullmatch(witness_sha))
            checks = (
                (t_sim != exp_sim, f"lean_witness_t_sim_mismatch:{cid}:witness={t_sim}:scored={exp_sim}"),
                (require_captured_p99 and (exp_p99 is None or exp_p99 <= 0), f"lean_captured_p99_missing:{cid}"),
                (exp_p99 is not None and t_p99 != exp_p99, f"lean_witness_p99_mismatch:{cid}:witness={t_p99}:captured={exp_p99}"),
                (t_p99 < exp_meas, f"lean_witness_p99_below_measured:{cid}"),
                (bool(require_program_dag_join and not exp_sha_ok), f"lean_scored_program_dag_sha256_missing:{cid}"),
                (bool(require_program_dag_join and not wit_sha_ok), f"lean_witness_program_dag_sha256_missing:{cid}"),
                (bool(require_program_dag_join and exp_sha_ok and wit_sha_ok and witness_sha != exp_sha), f"lean_witness_program_dag_mismatch:{cid}"),
            )
            for failed, msg in checks:
                if failed:
                    cert_valid = False
                    violations.append(msg)
        sim_inv, sil_inv = l_star > t_sim, l_star > t_p99
        if sim_inv:
            violations.append(f"lean_sim_inversion:{cid}:l_star={l_star}>t_sim={t_sim}")
        if sil_inv:
            violations.append(f"lean_silicon_p99_inversion:{cid}:l_star={l_star}>t_p99={t_p99}")
        if l_star <= 0 or t_sim <= 0 or t_p99 <= 0:
            cert_valid = False
            violations.append(f"lean_nonpositive_timing:{cid}")
        checked.append(LeanBoundCheckRow(cid, l_star, t_sim, t_p99, cert_checked, cert_valid, sim_inv, sil_inv))
    if expected_case_timings is not None and (missing := sorted(set(expected_case_timings) - seen_ids)):
        violations.append(f"lean_witness_missing_for_cases:{','.join(missing[:_MAX_MISSING_CASES_IN_MSG])}")
    return not violations, tuple(checked), tuple(violations)


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mindie_sim").is_dir() and (parent / "scripts").is_dir():
            return parent
    return here.parents[2]


def run_python_script(script: Path, args: Sequence[str], *, cwd: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-B", str(script), *args], cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout_s, check=False, **subprocess_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"onboarding_gate_timeout:{script.name}:{timeout_s}s") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_gate_exec_failed:{script.name}:{exc}") from exc


def _gate_script(repo_root: str | Path | None, rel_script: str, gate_name: str) -> tuple[Path, Path | None, GateCheckOutcome | None]:
    root = resolve_repo_root(repo_root)
    script = root / rel_script
    if not script.is_file():
        return root, None, GateCheckOutcome(gate_name, False, "script_missing", f"{rel_script} missing")
    return root, script, None


def _extend_ref_args(args: list[str], refs_root: str | Path | None, repo_map: Mapping[str, str | Path] | None) -> None:
    if refs_root is not None:
        args.extend(["--refs-root", str(Path(refs_root).resolve())])
    for name, path_val in sorted((repo_map or {}).items()):
        args.extend(["--repo-map", f"{name}={Path(path_val).resolve()}"])


def run_arch_governance_gate(
    repo_root: str | Path | None = None, *, strict: bool = True, refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None, timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> GateCheckOutcome:
    root, script, missing = _gate_script(repo_root, "scripts/check_arch_governance.py", "arch_governance")
    if missing is not None or script is None:
        return missing  # type: ignore[return-value]
    args = ["--root", str(root), *(["--strict"] if strict else [])]
    _extend_ref_args(args, refs_root, repo_map)
    done = run_python_script(script, args, cwd=root, timeout_s=timeout_s)
    text = (done.stdout or "") + ("\n" + done.stderr if done.stderr else "")
    m = _VIOLATION_COUNT_RE.search(text)
    cnt = int(m.group(1)) if m else (0 if done.returncode == 0 else 1)
    ok = done.returncode == 0 and cnt == 0
    return GateCheckOutcome("arch_governance", ok, "pass" if ok else "fail", "0 violations" if ok else "; ".join(ln for ln in text.splitlines() if ln.strip())[:_PREVIEW_CHAR_LIMIT], (("violations", cnt), ("returncode", done.returncode)))


def run_test_orthogonality_gate(
    repo_root: str | Path | None = None, *, strict: bool = True, base: str | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> GateCheckOutcome:
    root, script, missing = _gate_script(repo_root, "scripts/check_test_orthogonality.py", "test_orthogonality")
    if missing is not None or script is None:
        return missing  # type: ignore[return-value]
    args = ["--root", str(root), "--format", "json", *(["--base", str(base)] if base else (["--strict"] if strict else ["--base", "HEAD"]))]
    done = run_python_script(script, args, cwd=root, timeout_s=timeout_s)
    fails, detail = (0 if done.returncode == 0 else 1), "0 findings"
    try:
        bad = [f for f in (json.loads(done.stdout).get("findings") or []) if f.get("severity", "fail") == "fail"]
        fails, detail = len(bad), "; ".join(f"{f.get('path')}:{f.get('code')}" for f in bad[:_PREVIEW_LINES]) if bad else "0 findings"
    except ValueError:
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "orthogonality_gate_error").strip()
    ok = done.returncode == 0 and fails == 0
    return GateCheckOutcome("test_orthogonality", ok, "pass" if ok else "fail", detail, (("findings", fails), ("returncode", done.returncode)))


def run_template_provenance_gate(
    repo_root: str | Path | None = None, *, refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None, runtime: bool = True,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> GateCheckOutcome:
    root, script, missing = _gate_script(repo_root, "scripts/check_template_provenance.py", "template_provenance")
    if missing is not None or script is None:
        return missing  # type: ignore[return-value]
    args = ["--root", str(root), "--format", "json", *([] if runtime else ["--no-runtime"])]
    _extend_ref_args(args, refs_root, repo_map)
    done = run_python_script(script, args, cwd=root, timeout_s=timeout_s)
    try:
        parsed = json.loads(done.stdout)
        checked, viols = int(parsed.get("checked", 0)), list(parsed.get("violations") or [])
    except ValueError:
        return GateCheckOutcome("template_provenance", False, "fail", (done.stderr or done.stdout).strip())
    ok = done.returncode == 0 and not viols
    return GateCheckOutcome("template_provenance", ok, "pass" if ok else "fail", f"0 violations ({checked} refs)" if ok else "; ".join(str(v.get("reason")) for v in viols[:_PREVIEW_LINES]), (("checked", checked), ("violations", len(viols))))


def run_tail_cluster_diagnosis(
    residuals: Sequence[Mapping[str, Any]], *, repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT, lineage: Sequence[Mapping[str, Any]] = (),
    engine_receipts: Sequence[Mapping[str, Any]] = (), output_path: str | Path | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    cand = resolve_repo_root(repo_root)
    exec_root = cand if (cand / "mindie_sim" / "observations" / "multilevel.py").is_file() else resolve_repo_root(None)
    script = exec_root / "scripts" / "diagnose_tail_cluster.py"
    if not script.is_file():
        return {"schema": "mindie-sim.tail-cluster-diagnosis/1", "status": "diagnostic_script_unavailable", "cluster_count": 0, "clusters": []}
    with tempfile.TemporaryDirectory(prefix="msim-diag-") as tmp_dir:
        tmp = Path(tmp_dir)
        args = ["--residuals", str(write_atomic_json(tmp / "residuals.json", {"residuals": list(residuals)})), "--threshold-pct", str(float(threshold_pct))]
        if lineage:
            args.extend(["--lineage", str(write_atomic_json(tmp / "lineage.json", {"lineage": list(lineage)}))])
        if engine_receipts:
            args.extend(["--engine-receipts", str(write_atomic_json(tmp / "receipts.json", {"receipts": list(engine_receipts)}))])
        done = run_python_script(script, args, cwd=exec_root, timeout_s=timeout_s)
    if done.returncode not in (0, _DIAGNOSE_ESCALATED_EXIT) or not (done.stdout or "").strip():
        raise ModelingError(f"diagnose_tail_cluster_failed:rc={done.returncode}")
    report = json.loads(done.stdout)
    assert_sanitized_payload(report, field_name="tail_diagnosis.report")
    if output_path is not None:
        write_atomic_json(output_path, report)
    return report


def finalize_stage4_certification(
    *, schema: str, target_kind: Literal["sku", "operator"], target_id: str, sku: str,
    score_doc: Mapping[str, Any], lean_bounds_json: str | Path | Sequence[Mapping[str, Any]] | None,
    cap_pct: float, eff_atol: int, pre_gates: Sequence[GateCheckOutcome] = (),
    ape_gate_name: str, artifacts: Mapping[str, str], extra_warning_records: Sequence[Mapping[str, Any] | None] = (),
    capture_root: str | Path | None = None, require_capture_files: bool = False,
    repo_root: str | Path | None = None, run_repo_gates: bool = True, orthogonality_base: str | None = None,
    out_receipt: str | Path | None = None, diagnosis_out: str | Path | None = None,
) -> tuple[bool, str, dict[str, Any], dict[str, Any] | None]:
    """Shared Stage-4 gate verification (SKU/capture provenance, APE tolerance, certified lower-bound non-inversion, repo gates, tail diagnosis, SHA-256 receipt)."""
    residuals = list(score_doc.get("residuals") or ())
    score_sku = str(score_doc.get("sku") or "").strip()
    plan_obj = score_doc.get("plan") if isinstance(score_doc.get("plan"), Mapping) else None
    plan_sku = str(plan_obj.get("sku") or "").strip() if plan_obj is not None else ""
    foreign_rows = [str(r.get("case_id") or "") for r in residuals if str(r.get("sku") or "").strip() != sku]
    sku_ok = (score_sku == sku) and (not plan_sku or plan_sku == sku) and not foreign_rows
    sku_detail = (
        f"sku={sku}"
        if sku_ok
        else f"target_sku_mismatch:expected={sku}:score_sku={score_sku or 'missing'}:foreign_rows={len(foreign_rows)}"
    )
    heldout = [
        r for r in residuals
        if r.get("split") == "heldout" and str(r.get("status") or "ok") == "ok" and str(r.get("sku") or "").strip() == sku
    ]
    failing = [
        r for r in heldout
        if exceeds_physical_tolerance(
            int(r.get("predicted_ns") or 0), int(r.get("measured_ns") or 0),
            threshold_pct=cap_pct, atol_ns=eff_atol,
            allow_signed=(str(r.get("boundary_kind") or "") == "paired_rung_delta"),
        )
    ]
    ape_ok = sku_ok and bool(score_doc.get("ok")) and bool(heldout) and not failing
    gates: list[GateCheckOutcome] = [
        *pre_gates,
        GateCheckOutcome("target_sku_provenance", sku_ok, "pass" if sku_ok else "fail", sku_detail, (("expected_sku", sku), ("score_sku", score_sku), ("foreign_row_count", len(foreign_rows)))),
        GateCheckOutcome(ape_gate_name, ape_ok, "pass" if ape_ok else "fail", f"{len(heldout) - len(failing)}/{len(heldout)} heldout within max({cap_pct}%, {eff_atol}ns)", (("heldout_count", len(heldout)), ("failing_count", len(failing)))),
    ]
    eff_artifacts = dict(artifacts)
    if require_capture_files or capture_root is not None:
        cap_ok, cap_cnt, cap_detail, cap_bundle_sha = _verify_residual_capture_files(residuals, capture_root, require_capture_files)
        gates.append(GateCheckOutcome("capture_files_sha256", cap_ok, "pass" if cap_ok else "fail", cap_detail, (("verified_count", cap_cnt), ("required", bool(require_capture_files)))))
        if cap_bundle_sha:
            eff_artifacts["verified_captures_sha256"] = cap_bundle_sha
    lean_rows = list(read_bounded_json(lean_bounds_json) if isinstance(lean_bounds_json, (str, Path)) else (lean_bounds_json or ()))
    exp_timings = {str(r["case_id"]): (int(r["predicted_ns"]), int(r["measured_ns"]), r.get("measured_p99_ns"), r.get("program_dag_sha256")) for r in heldout}
    lean_ok, lean_checked, lean_viols = check_lean_non_inversion(lean_rows, expected_case_timings=exp_timings)
    gates.append(GateCheckOutcome("lean_optimality_non_inversion", lean_ok, "pass" if lean_ok else "fail", "0 inversions" if lean_ok else "; ".join(lean_viols[:_PREVIEW_LINES]), (("checked", len(lean_checked)), ("violations", len(lean_viols)))))
    if run_repo_gates:
        gates.append(run_arch_governance_gate(repo_root=repo_root, strict=True))
        gates.append(run_test_orthogonality_gate(repo_root=repo_root, strict=(orthogonality_base is None), base=orthogonality_base))
    diagnosis = run_tail_cluster_diagnosis(residuals, repo_root=repo_root, threshold_pct=cap_pct, output_path=diagnosis_out) if failing else None
    all_ok = all(g.passed for g in gates)
    acq_mode = str(score_doc.get("acquisition_mode") or "offline_reverify")
    status = ("silicon_certified" if acq_mode == "live_silicon" else "offline_reverify") if all_ok else "failed"
    rec_u, rec_w = collect_receipt_warnings(score_doc, *extra_warning_records)
    receipt = seal_onboarding_receipt(
        schema=schema, target_kind=target_kind, target_id=target_id, sku=sku, status=status, ok=all_ok,
        gates=gates, artifacts=eff_artifacts,
        summary={"heldout_count": len(heldout), "failing_count": len(failing), "threshold_pct": cap_pct, "atol_ns": eff_atol, "acquisition_mode": acq_mode},
        receipt_unconfirmed=rec_u, receipt_warning=rec_w,
    ).to_dict()
    if out_receipt is not None:
        write_atomic_json(out_receipt, receipt)
    return all_ok, status, receipt, diagnosis
