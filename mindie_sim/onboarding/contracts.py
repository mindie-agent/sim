"""Immutable contracts, timeout budgets, sanitization, and Lean/receipt checks for onboarding.

[Role] Define shared immutable dataclasses, canonical domain constants,
  sanitized ``capture://`` provenance helpers, timeout-arithmetic budgets,
  SHA-256 receipt sealing/verification, and Lean 4 ``OptimalityCertificate``
  non-inversion checks for SKU and operator onboarding pipelines.
[Invariants] Every timeout budget enforces
  ``outer_timeout_s >= sum(inner_budgets) + cleanup_grace_s``. All emitted
  payloads fail closed on private host paths, IPv4 quads, usernames, or
  container digests. Lean checks fail closed on any ``L*_Lean > T_sim`` or
  ``L*_Lean > T_silicon_p99`` inversion or invalid certificate witness.
[Timeouts & Locks] Bounded 50 MiB JSON reads and atomic ``os.replace`` writes
  with explicit temp-file cleanup; ``TimeoutError`` always propagates before
  ``OSError``. Zero locks or device contexts.
[Side-Effect Policy] Pure validation and sealing helpers; filesystem mutation
  occurs only in explicit ``write_atomic_json`` calls.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Literal

from mindie_sim.observations.multilevel_stream import (
    SHA256_HEX_LENGTH,
    sanitize_capture_uri,
    validate_provenance_text,
)
from mindie_sim.schema.common import ModelingError
from .tiling_axes import (
    AbiDivisibilityRule,
    TilingAxes,
    default_topk_abi_rule,
    validate_abi,
)

SKU_ONBOARDING_RECEIPT_SCHEMA = "mindie-sim.sku-onboarding-receipt/1"
OPERATOR_ONBOARDING_RECEIPT_SCHEMA = "mindie-sim.operator-onboarding-receipt/1"
CLIFF_PLAN_SCHEMA = "mindie-sim.cliff-bench-plan/2"
SOC_PROBE_SCHEMA = "mindie-sim.soc-probe/1"
TRAIN_OBSERVATION_MANIFEST_SCHEMA = "mindie-sim.train-observation-manifest/1"
VALID_CLOCK_OPERATORS: tuple[str, ...] = ("busy_union", "elapsed_span")
DEFAULT_CLOCK_OPERATOR = "elapsed_span"
FORBIDDEN_CALLER_PREDICTION_KEYS: frozenset[str] = frozenset(
    {
        "predicted_ns",
        "predicted_ms",
        "predicted_us",
        "predicted_s",
        "predicted_duration",
    }
)

DEFAULT_HELDOUT_APE_THRESHOLD_PCT = 10.0
ANCHOR_CONTEXT_GRID: tuple[int, ...] = (
    512, 2048, 8192, 32768, 131072, 262144, 409600,
)
OFF_GRID_CLIFF_HELDOUT_GRID: tuple[int, ...] = (
    768, 2049, 4097, 24576, 98304, 200000, 350000, 400000,
)
CONTEXT_MIN_TOKENS = 512
CONTEXT_MAX_TOKENS = 409600
DEFAULT_SPARSE_TOPK = 2048
COLD_RING_L2_MULTIPLIER = 3
PERCENT_SCALE = 100.0
NS_PER_MS = 1_000_000
NS_PER_US = 1000
MIB_BYTES = 1024 * 1024
GIB_BYTES = 1024 * 1024 * 1024
MAX_READ_JSON_BYTES = 50 * MIB_BYTES
DEFAULT_DEVICE_VRAM_BUDGET_GIB = 8
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
DEFAULT_OUTER_TIMEOUT_S = (
    DEFAULT_L0_BUDGET_S + DEFAULT_FABRIC_BUDGET_S + DEFAULT_CLEANUP_GRACE_S
)

_IPV4_QUAD_RE = re.compile(r"(?<!\d)\d{1,3}(?:[._-]\d{1,3}){3}(?!\d)")
_SAFE_SOC_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
_SAFE_ALIAS_RE = re.compile(r"[^a-z0-9._-]+")
_EMAIL_OR_USER_HOST_RE = re.compile(
    r"(?:^|[\s\"'(<:,=])[A-Za-z_][A-Za-z0-9._%+-]*@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?=$|[\s\"')>,;])"
    r"|[A-Za-z_][A-Za-z0-9._-]*@(?:\d{1,3}\.){3}\d{1,3}"
)
_HOME_ENV_RE = re.compile(r"\$\{?HOME\}?")
_GENERIC_ALIASES = frozenset({"a2", "a3", "a5", "host", "default", "root", "user"})
_SUBPROCESS_NO_WINDOW = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


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
        if type(self.outer_timeout_s) is not int or self.outer_timeout_s < 1:
            raise ModelingError(f"timeout_budget_outer_invalid:{self.outer_timeout_s}")
        required = self.minimum_outer_timeout_s
        if self.outer_timeout_s < required:
            raise ModelingError(
                f"timeout_budget_outer_too_small:outer={self.outer_timeout_s}:required={required}"
            )

    @property
    def total_inner_s(self) -> int:
        return sum(budget for _, budget in self.inner_budgets_s)

    @property
    def minimum_outer_timeout_s(self) -> int:
        return self.total_inner_s + self.cleanup_grace_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_budgets_s": {name: value for name, value in self.inner_budgets_s},
            "total_inner_s": self.total_inner_s,
            "cleanup_grace_s": self.cleanup_grace_s,
            "outer_timeout_s": self.outer_timeout_s,
        }


@dataclass(frozen=True, slots=True)
class SiliconProfileConflict:
    """Single mismatch between observed silicon truth and declared SKU profile."""

    field: str
    observed: Any
    declared: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SocProbeRecord:
    """Sanitized hardware probe facts from ``aclrtGetSocName``, CANN INI, and ``npu-smi``."""

    soc_name: str
    arch_preset: str
    architecture: str
    aic_count: int
    aiv_count: int
    aicore_freq_mhz: int
    hbm_bytes: int
    l2_bytes: int
    ub_bytes: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    dies_per_package: int
    ranks_per_node: int
    cann_version: str
    endpoint_alias: str
    capture_uri: str
    probe_sha256: str
    fabric_channels: tuple[tuple[str, float], ...] = ()
    topology_preset: str = ""

    def __post_init__(self) -> None:
        if not self.soc_name or not _SAFE_SOC_NAME_RE.fullmatch(self.soc_name.strip()):
            raise ModelingError("soc_probe_name_invalid")
        if self.aic_count < 1:
            raise ModelingError(f"soc_probe_aic_invalid:{self.aic_count}")
        if self.aiv_count < 0:
            raise ModelingError(f"soc_probe_aiv_invalid:{self.aiv_count}")
        if self.dies_per_package < 1 or self.ranks_per_node < 1:
            raise ModelingError("soc_probe_topology_invalid")
        if len(self.probe_sha256) != SHA256_HEX_LENGTH:
            raise ModelingError("soc_probe_sha256_invalid")
        sanitize_capture_uri(self.capture_uri)
        validate_provenance_text(self.endpoint_alias, field_name="endpoint_alias")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["fabric_channels"] = {name: val for name, val in self.fabric_channels}
        return raw


@dataclass(frozen=True, slots=True)
class CliffPointSpec:
    """Single deterministic anchor, 2D cliff, or ABI-rejection benchmark point."""

    point_id: str
    split: Literal["train", "heldout", "abi_rejection"]
    grid_kind: Literal["anchor", "off_grid_cliff", "boundary_cliff", "cross_sample_2d", "abi_rejection"]
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

    def __post_init__(self) -> None:
        if not self.point_id or not self.axis or type(self.value) is not int or self.value < 1:
            raise ModelingError(f"cliff_point_invalid:{self.point_id}:{self.axis}:{self.value}")
        if self.split not in ("train", "heldout", "abi_rejection"):
            raise ModelingError(f"cliff_point_split_invalid:{self.split}")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ModelingError(f"cliff_point_batch_invalid:{self.batch_size}")
        if type(self.query_tokens) is not int or self.query_tokens < 1:
            raise ModelingError(f"cliff_point_query_invalid:{self.query_tokens}")
        if self.clock_operator not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"cliff_point_clock_operator_invalid:{self.clock_operator}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperatorCliffParameters:
    """Declared physical parameters driving operator cliff-plan synthesis."""

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

    def __post_init__(self) -> None:
        if not self.operator_id or not self.family:
            raise ModelingError("operator_cliff_identity_missing")
        if type(self.tile_dim) is not int or self.tile_dim < 2:
            raise ModelingError(f"operator_cliff_tile_invalid:{self.tile_dim}")
        if type(self.core_count) is not int or self.core_count < 1:
            raise ModelingError(f"operator_cliff_core_count_invalid:{self.core_count}")
        if type(self.l2_bytes) is not int or self.l2_bytes < 1:
            raise ModelingError(f"operator_cliff_l2_invalid:{self.l2_bytes}")
        if type(self.bytes_per_token) is not int or self.bytes_per_token < 0:
            raise ModelingError(f"operator_cliff_bytes_per_token_invalid:{self.bytes_per_token}")
        if self.topk_threshold is not None and (
            type(self.topk_threshold) is not int or self.topk_threshold < 2
        ):
            raise ModelingError(f"operator_cliff_topk_invalid:{self.topk_threshold}")
        if self.context_min < 1 or self.context_max < self.context_min:
            raise ModelingError("operator_cliff_context_domain_invalid")


@dataclass(frozen=True, slots=True)
class CliffBenchPlan:
    """SHA-256-sealed disjoint train + 2D cliff heldout + ABI-rejection bench plan."""

    schema: str
    plan_id: str
    target_id: str
    sku: str
    train_lineage_root: str
    heldout_lineage_root: str
    cold_ring_l2_multiplier: int
    train_points: tuple[CliffPointSpec, ...]
    heldout_points: tuple[CliffPointSpec, ...]
    plan_sha256: str
    abi_rejection_points: tuple[CliffPointSpec, ...] = ()
    tiling_axes: TilingAxes | None = None

    def __post_init__(self) -> None:
        if self.schema != CLIFF_PLAN_SCHEMA:
            raise ModelingError(f"cliff_plan_schema_invalid:{self.schema}")
        if not self.train_lineage_root or not self.heldout_lineage_root:
            raise ModelingError("cliff_plan_lineage_root_missing")
        if self.train_lineage_root == self.heldout_lineage_root:
            raise ModelingError("cliff_plan_train_heldout_lineage_overlap")
        if self.cold_ring_l2_multiplier < COLD_RING_L2_MULTIPLIER:
            raise ModelingError(
                f"cliff_plan_cold_ring_too_small:{self.cold_ring_l2_multiplier}"
            )
        train_keys = {
            (pt.axis, pt.value, pt.batch_size, pt.query_tokens, pt.parallel_topology)
            for pt in self.train_points
        }
        heldout_keys = {
            (pt.axis, pt.value, pt.batch_size, pt.query_tokens, pt.parallel_topology)
            for pt in self.heldout_points
        }
        overlap = sorted(train_keys & heldout_keys)
        if overlap:
            raise ModelingError(f"cliff_plan_train_heldout_point_overlap:{overlap[0]}")
        train_ctx = {pt.value for pt in self.train_points if pt.axis == "context_len"}
        heldout_ctx = {pt.value for pt in self.heldout_points if pt.axis == "context_len"}
        ctx_overlap = sorted(train_ctx & heldout_ctx)
        if ctx_overlap:
            raise ModelingError(f"cliff_plan_train_heldout_context_overlap:{ctx_overlap[0]}")
        if len(self.plan_sha256) != SHA256_HEX_LENGTH:
            raise ModelingError("cliff_plan_sha256_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "target_id": self.target_id,
            "sku": self.sku,
            "train_lineage_root": self.train_lineage_root,
            "heldout_lineage_root": self.heldout_lineage_root,
            "cold_ring_l2_multiplier": self.cold_ring_l2_multiplier,
            "train_points": [pt.to_dict() for pt in self.train_points],
            "heldout_points": [pt.to_dict() for pt in self.heldout_points],
            "abi_rejection_points": [pt.to_dict() for pt in self.abi_rejection_points],
            "tiling_axes": self.tiling_axes.to_dict() if self.tiling_axes is not None else None,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class LeanBoundCheckRow:
    """Result of verifying Lean 4 lower-bound non-inversion for one benchmark case."""

    case_id: str
    l_star_lean_ns: int
    t_sim_ns: int
    t_silicon_p99_ns: int
    certificate_checked: bool
    certificate_valid: bool
    sim_inverted: bool
    silicon_inverted: bool

    @property
    def passed(self) -> bool:
        return (
            self.certificate_valid
            and not self.sim_inverted
            and not self.silicon_inverted
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True, slots=True)
class GateCheckOutcome:
    """Single deterministic gate outcome embedded in an onboarding receipt."""

    name: str
    passed: bool
    status: str
    detail: str = ""
    metrics: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "detail": self.detail,
            "metrics": {key: val for key, val in self.metrics},
        }


@dataclass(frozen=True, slots=True)
class OnboardingReceipt:
    """Machine-verifiable SHA-256 onboarding certification receipt."""

    schema: str
    kind: Literal["sku", "operator"]
    target_id: str
    sku: str
    status: Literal["certified", "failed"]
    ok: bool
    created_utc: str
    gates: tuple[GateCheckOutcome, ...]
    artifacts: tuple[tuple[str, str], ...]
    diagnosis_report: Mapping[str, Any] | None
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "target_id": self.target_id,
            "sku": self.sku,
            "status": self.status,
            "ok": self.ok,
            "created_utc": self.created_utc,
            "gates": [gate.to_dict() for gate in self.gates],
            "artifacts": {key: val for key, val in self.artifacts},
            "diagnosis_report": dict(self.diagnosis_report) if self.diagnosis_report else None,
            "receipt_sha256": self.receipt_sha256,
        }


def utc_now_iso() -> str:
    """Return canonical UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize ``payload`` deterministically with ASCII-safe canonical JSON."""
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return text.encode("utf-8")


def compute_receipt_sha256(
    payload: Mapping[str, Any], *, digest_field: str = "receipt_sha256"
) -> str:
    """Compute canonical SHA-256 over all receipt fields except ``digest_field``."""
    body = {key: value for key, value in payload.items() if key != digest_field}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def verify_receipt_sha256(
    payload: Mapping[str, Any], *, digest_field: str = "receipt_sha256"
) -> bool:
    """Return ``True`` iff the stored SHA-256 matches canonical recomputation."""
    expected = payload.get(digest_field)
    if not isinstance(expected, str) or len(expected) != SHA256_HEX_LENGTH:
        return False
    return expected.lower() == compute_receipt_sha256(payload, digest_field=digest_field)


def sanitize_public_alias(alias: str, fallback_tag: str) -> str:
    """Return a clean public alias with zero raw IPv4 quads, paths, or usernames."""
    cleaned = _IPV4_QUAD_RE.sub("", str(alias or "").lower())
    cleaned = _EMAIL_OR_USER_HOST_RE.sub("", cleaned)
    cleaned = _HOME_ENV_RE.sub("", cleaned)
    cleaned = _SAFE_ALIAS_RE.sub("-", cleaned).strip("._-")
    if not cleaned or cleaned in _GENERIC_ALIASES:
        fallback = _SAFE_ALIAS_RE.sub("-", str(fallback_tag or "target").lower()).strip("._-")
        return f"lab-{fallback or 'target'}"
    return cleaned


def build_capture_uri(
    alias: str, campaign_id: str, content_bytes: bytes
) -> tuple[str, str]:
    """Build a sanitized ``capture://<alias>/<campaign>/<sha256>`` URI and hex digest."""
    clean_alias = sanitize_public_alias(alias, "capture")
    clean_campaign = _SAFE_ALIAS_RE.sub("-", str(campaign_id or "onboard").lower()).strip("._-")
    clean_campaign = clean_campaign or "onboard"
    digest = hashlib.sha256(content_bytes).hexdigest()
    uri = sanitize_capture_uri(f"capture://{clean_alias}/{clean_campaign}/{digest}")
    return uri, digest


def assert_sanitized_payload(payload: Any, *, field_name: str = "payload") -> None:
    """Fail closed if any nested string leaks private host paths, IPs, or user emails."""
    if isinstance(payload, str):
        validate_provenance_text(payload, field_name=field_name)
        if _HOME_ENV_RE.search(payload) or _EMAIL_OR_USER_HOST_RE.search(payload):
            raise ModelingError(f"private_identity_rejected:{field_name}")
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_sanitized_payload(str(key), field_name=f"{field_name}.key")
            assert_sanitized_payload(value, field_name=f"{field_name}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            assert_sanitized_payload(item, field_name=f"{field_name}[{idx}]")


def read_bounded_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file bounded to ``MAX_READ_JSON_BYTES``."""
    target = Path(path)
    try:
        size = target.stat().st_size
        if size > MAX_READ_JSON_BYTES:
            raise ModelingError(f"onboarding_json_exceeds_50_mib:{target.name}")
        raw = target.read_text(encoding="utf-8")
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_json_read_failed:{target.name}:{exc}") from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ModelingError(f"onboarding_json_invalid:{target.name}:{exc}") from exc


def write_atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write formatted UTF-8/LF JSON atomically via sibling temp file and ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    tmp_path: Path | None = None
    try:
        fd, raw_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        tmp_path = Path(raw_name)
        try:
            os.write(fd, rendered)
        finally:
            os.close(fd)
        os.replace(tmp_path, target)
        tmp_path = None
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_json_write_failed:{target.name}:{exc}") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def subprocess_window_kwargs() -> dict[str, Any]:
    """Return windowless ``creationflags`` on Windows and empty dict on POSIX."""
    return {"creationflags": _SUBPROCESS_NO_WINDOW} if _SUBPROCESS_NO_WINDOW else {}


from ._lean_and_receipts import (  # noqa: E402
    check_lean_non_inversion,
    seal_onboarding_receipt,
)


__all__ = [
    "ANCHOR_CONTEXT_GRID",
    "CLIFF_PLAN_SCHEMA",
    "COLD_RING_L2_MULTIPLIER",
    "CONTEXT_MAX_TOKENS",
    "CONTEXT_MIN_TOKENS",
    "DEFAULT_CLEANUP_GRACE_S",
    "DEFAULT_CLOCK_OPERATOR",
    "DEFAULT_DEVICE_VRAM_BUDGET_GIB",
    "DEFAULT_FABRIC_BUDGET_S",
    "DEFAULT_GATE_SUBPROCESS_TIMEOUT_S",
    "DEFAULT_HELDOUT_APE_THRESHOLD_PCT",
    "DEFAULT_L0_BUDGET_S",
    "DEFAULT_OUTER_TIMEOUT_S",
    "DEFAULT_PROBE_TIMEOUT_S",
    "DEFAULT_SPARSE_TOPK",
    "DEFAULT_SSH_CONNECT_TIMEOUT_S",
    "DISPLAY_ROUND_DIGITS",
    "FORBIDDEN_CALLER_PREDICTION_KEYS",
    "GIB_BYTES",
    "HEX_COMMIT_LEN",
    "MAX_CALIBRATED_CV_PCT",
    "MAX_READ_JSON_BYTES",
    "MIB_BYTES",
    "NS_PER_MS",
    "NS_PER_US",
    "OFF_GRID_CLIFF_HELDOUT_GRID",
    "OPERATOR_ONBOARDING_RECEIPT_SCHEMA",
    "PERCENT_SCALE",
    "SHORT_DIGEST_LEN",
    "SKU_ONBOARDING_RECEIPT_SCHEMA",
    "SOC_PROBE_SCHEMA",
    "TRAIN_OBSERVATION_MANIFEST_SCHEMA",
    "VALID_CLOCK_OPERATORS",
    "AbiDivisibilityRule",
    "CliffBenchPlan",
    "CliffPointSpec",
    "GateCheckOutcome",
    "LeanBoundCheckRow",
    "OnboardingReceipt",
    "OperatorCliffParameters",
    "SiliconProfileConflict",
    "SocProbeRecord",
    "TilingAxes",
    "TimeoutBudget",
    "assert_sanitized_payload",
    "build_capture_uri",
    "canonical_json_bytes",
    "check_lean_non_inversion",
    "compute_receipt_sha256",
    "default_topk_abi_rule",
    "read_bounded_json",
    "sanitize_public_alias",
    "seal_onboarding_receipt",
    "subprocess_window_kwargs",
    "utc_now_iso",
    "validate_abi",
    "verify_receipt_sha256",
    "write_atomic_json",
]
