"""Version-pinned C7 modeling interface: provider protocol, prediction, alignment.

[Role]
  Define the versioned provider boundary and the three explicitly separated C7
  values: ``ideal_hardware_lower_bound_ms``, ``model_predicted_ms`` and
  ``measured_device_ms``.  Input metadata and the local roofline formula live
  in ``modeling_inputs``; this module accepts only an injected, identity-bearing
  provider for external predictions and never guesses a provider API.

[Invariants]
  * Missing metadata inputs or unavailable workload data return
    ``unmodeled_reason``; model-derived numeric fields stay ``None``.
  * A prediction requires a real provider name, a real version, the exact
    adapter protocol version, the request input signature and non-empty
    evidence.  Provider rejections preserve a stable reason.
  * Placeholder versions, protocol mismatches, signature mismatches and
    provider exceptions are unmodeled; no local roofline value is relabeled as
    an upstream prediction.
  * External provider efficiency is applied by the provider only; the local
    ideal bound uses raw physical peaks from ``modeling_inputs``.

[Timeouts & Locks]
  Pure computation.  No locks, sleeps, I/O, subprocesses, network or caches.

[Side-Effect Policy]
  Read-only ``importlib.util.find_spec`` probe for reporting install state.  The
  probe never imports or executes msmodeling and never claims it is verified.
"""
from __future__ import annotations

import importlib.util
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .modeling_inputs import (
    IDEAL_BOUND_FORMULA_VERSION,
    HardwareSpec,
    IdealLowerBound,
    KernelInputs,
    estimate_ideal_hardware_lower_bound,
    ideal_hardware_lower_bound,
)
from .sku_profile import (
    AMBIGUOUS_SKU_REASON,
    BUILTIN_SKU_CATALOG,
    DeviceCapabilityProfile,
    is_ambiguous_family_token,
    resolve_device_capability_profile,
)

ADAPTER_PROTOCOL_VERSION = "mindie-msmodeling-adapter/1"
MSMODELING_MODULE = "msmodeling"

MODELED = "modeled"
UNMODELED = "unmodeled"

_VERSION_RE = re.compile(
    r"^(?:v?\d+\.\d+\.\d+(?:[.+-][0-9A-Za-z.-]+)*|git:[0-9a-fA-F]{7,40})$"
)
_PLACEHOLDER_VERSIONS = {
    "",
    "unknown",
    "unverified",
    "unavailable",
    "none",
    "null",
    "n/a",
    "na",
    "dev",
    "development",
    "latest",
    "0.0.0",
}
_MISSING = object()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _real_version(version: Any) -> bool:
    if not isinstance(version, str):
        return False
    text = version.strip()
    return text.lower() not in _PLACEHOLDER_VERSIONS and bool(_VERSION_RE.match(text))


@dataclass(frozen=True)
class ProviderIdentity:
    provider_name: str
    provider_version: str
    adapter_protocol_version: str = ADAPTER_PROTOCOL_VERSION

class ProviderUnmodeledError(ValueError):
    """Provider rejected a request with a stable, caller-safe reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)

@dataclass(frozen=True)
class ProviderPrediction:
    predicted_ms: float
    model_assumptions: str
    input_signature: str
    evidence: tuple[str, ...] = ()

@dataclass(frozen=True)
class ModelingRequest:
    inputs: KernelInputs
    measured_device_ms: float | None = None
    measured_source: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.inputs, Mapping):
            object.__setattr__(self, "inputs", KernelInputs.from_mapping(self.inputs))
        if isinstance(self.evidence, str):
            object.__setattr__(self, "evidence", (self.evidence,))
        else:
            object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def input_signature(self) -> str:
        return self.inputs.input_signature()

@runtime_checkable
class ModelingProvider(Protocol):
    """Fixed adapter protocol; implementations must provide all three members."""

    @property
    def identity(self) -> ProviderIdentity: ...

    def available(self) -> bool: ...

    def predict(self, request: ModelingRequest) -> ProviderPrediction: ...

@dataclass(frozen=True)
class ModelPrediction:
    status: str
    predicted_ms: float | None
    provider_name: str | None
    provider_version: str | None
    adapter_protocol_version: str | None
    input_signature: str | None
    model_assumptions: str | None
    evidence: tuple[str, ...]
    unmodeled_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_predicted_ms": self.predicted_ms,
            "model_provider": (
                f"{self.provider_name}@{self.provider_version}"
                if self.provider_name and self.provider_version
                else None
            ),
            "adapter_protocol_version": self.adapter_protocol_version,
            "input_signature": self.input_signature,
            "model_assumptions": self.model_assumptions,
            "evidence": list(self.evidence),
            "unmodeled_reason": self.unmodeled_reason,
        }

def _unmodeled_prediction(reason: str, *, signature: str | None = None) -> ModelPrediction:
    return ModelPrediction(
        status=UNMODELED,
        predicted_ms=None,
        provider_name=None,
        provider_version=None,
        adapter_protocol_version=None,
        input_signature=signature,
        model_assumptions=None,
        evidence=(),
        unmodeled_reason=reason,
    )

def _coerce_request(value: ModelingRequest | KernelInputs | Mapping[str, Any]) -> ModelingRequest:
    if isinstance(value, ModelingRequest):
        return value
    if isinstance(value, KernelInputs):
        return ModelingRequest(inputs=value)
    if isinstance(value, Mapping):
        if "inputs" in value:
            return ModelingRequest(
                inputs=value["inputs"],
                measured_device_ms=value.get("measured_device_ms"),
                measured_source=value.get("measured_source"),
                evidence=value.get("evidence", ()),
            )
        return ModelingRequest(inputs=KernelInputs.from_mapping(value))
    raise TypeError("request must be ModelingRequest, KernelInputs or a mapping")

def _provider_identity(provider: Any) -> tuple[ProviderIdentity | None, str | None]:
    identity = getattr(provider, "identity", None)
    if isinstance(identity, Mapping):
        try:
            identity = ProviderIdentity(**dict(identity))
        except (TypeError, ValueError):
            return None, "provider_identity_invalid"
    if not isinstance(identity, ProviderIdentity):
        return None, "provider_identity_missing"
    if not isinstance(identity.provider_name, str) or not identity.provider_name.strip():
        return None, "provider_name_missing"
    if not _real_version(identity.provider_version):
        return None, "provider_version_not_real"
    if identity.adapter_protocol_version != ADAPTER_PROTOCOL_VERSION:
        return None, (
            "adapter_protocol_mismatch:"
            f"expected={ADAPTER_PROTOCOL_VERSION},got={identity.adapter_protocol_version}"
        )
    return identity, None

def _provider_available(provider: Any) -> tuple[bool | None, str | None]:
    available = getattr(provider, "available", None)
    if available is None:
        return None, "provider_availability_unknown"
    try:
        result = available() if callable(available) else available
    except Exception as exc:  # noqa: BLE001 - provider boundary is untrusted
        return None, f"provider_availability_error:{type(exc).__name__}"
    if not isinstance(result, bool):
        return None, "provider_availability_invalid"
    return result, None

def _prediction_field(prediction: Any, name: str) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(name, _MISSING)
    return getattr(prediction, name, _MISSING)

def _prediction_evidence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if value is None:
        return ()
    try:
        items = tuple(str(item).strip() for item in value)
    except TypeError:
        return ()
    return tuple(item for item in items if item)

def predict_with_provider(
    request: ModelingRequest | KernelInputs | Mapping[str, Any],
    provider: Any = None,
) -> ModelPrediction:
    """Call an injected versioned provider, or explain why no prediction exists."""
    try:
        req = _coerce_request(request)
    except (TypeError, ValueError) as exc:
        return _unmodeled_prediction(f"invalid_request:{type(exc).__name__}")
    signature = req.input_signature
    missing = req.inputs.missing_metadata_inputs()
    if missing:
        return _unmodeled_prediction("missing_inputs:" + ",".join(missing), signature=signature)
    if provider is None:
        return _unmodeled_prediction("msmodeling_provider_not_configured", signature=signature)
    identity, identity_error = _provider_identity(provider)
    if identity_error is not None:
        return _unmodeled_prediction(identity_error, signature=signature)
    assert identity is not None
    available, availability_error = _provider_available(provider)
    if availability_error is not None:
        return _unmodeled_prediction(availability_error, signature=signature)
    if not available:
        return _unmodeled_prediction("provider_unavailable", signature=signature)
    try:
        raw = provider.predict(req)
    except ProviderUnmodeledError as exc:
        return _unmodeled_prediction(exc.reason, signature=signature)
    except Exception as exc:  # noqa: BLE001 - provider boundary is untrusted
        return _unmodeled_prediction(f"provider_error:{type(exc).__name__}", signature=signature)
    predicted = _prediction_field(raw, "predicted_ms")
    assumptions = _prediction_field(raw, "model_assumptions")
    returned_signature = _prediction_field(raw, "input_signature")
    evidence = _prediction_evidence(_prediction_field(raw, "evidence"))
    if not _finite(predicted) or float(predicted) <= 0:
        return _unmodeled_prediction("provider_prediction_invalid", signature=signature)
    if not isinstance(assumptions, str) or not assumptions.strip():
        return _unmodeled_prediction("missing_model_assumptions", signature=signature)
    if returned_signature != signature:
        return _unmodeled_prediction("input_signature_mismatch", signature=signature)
    if not evidence:
        return _unmodeled_prediction("missing_provider_evidence", signature=signature)
    return ModelPrediction(
        status=MODELED,
        predicted_ms=float(predicted),
        provider_name=identity.provider_name,
        provider_version=identity.provider_version,
        adapter_protocol_version=identity.adapter_protocol_version,
        input_signature=signature,
        model_assumptions=assumptions.strip(),
        evidence=evidence,
        unmodeled_reason=None,
    )

@dataclass(frozen=True)
class ModelingAlignment:
    status: str
    measured_device_ms: float | None
    ideal_hardware_lower_bound_ms: float | None
    model_predicted_ms: float | None
    model_provider: str | None
    model_provider_version: str | None
    adapter_protocol_version: str | None
    input_signature: str | None
    model_assumptions: str | None
    evidence: tuple[str, ...]
    unmodeled_reason: str | None
    measurement_limitation: str | None = None

    def to_c7_alignment(self) -> dict[str, Any]:
        # These ratios compare a measured duration with declared profile/model
        # values. They do not measure an optimization or verify a board's peaks.
        measured = self.measured_device_ms
        ideal = self.ideal_hardware_lower_bound_ms
        predicted = self.model_predicted_ms
        return {
            "status": MODELED if self.status == MODELED else UNMODELED,
            "ideal_hardware_lower_bound_ms": self.ideal_hardware_lower_bound_ms,
            "model_predicted_ms": self.model_predicted_ms,
            "measured_device_ms": self.measured_device_ms,
            "model_provider": self.model_provider,
            "model_assumptions": self.model_assumptions,
            "unmodeled_reason": self.unmodeled_reason,
            "input_signature": self.input_signature,
            "model_evidence": list(self.evidence),
            "measurement_limitation": self.measurement_limitation,
            "time_comparison": {
                "claim_scope": "declared_hardware_profile",
                "descriptive_only": True,
                "physical_hardware_mapping_verified": False,
                "measured_over_ideal": (
                    measured / ideal
                    if measured is not None and ideal is not None and ideal > 0
                    else None
                ),
                "measured_over_prediction": (
                    measured / predicted
                    if measured is not None and predicted is not None and predicted > 0
                    else None
                ),
            },
        }

    as_c7_dict = to_c7_alignment

def _coerce_measured(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if not _finite(value) or float(value) < 0:
        return None, "invalid_measured_device_ms"
    return float(value), None

def align_modeling(
    request: ModelingRequest | KernelInputs | Mapping[str, Any],
    provider: Any = None,
    *,
    measured_device_ms: float | None = None,
    measured_source: str | None = None,
) -> ModelingAlignment:
    """Produce the C7 three-way alignment; partial evidence is never invented."""
    try:
        req = _coerce_request(request)
    except (TypeError, ValueError) as exc:
        return ModelingAlignment(
            status=UNMODELED,
            measured_device_ms=None,
            ideal_hardware_lower_bound_ms=None,
            model_predicted_ms=None,
            model_provider=None,
            model_provider_version=None,
            adapter_protocol_version=None,
            input_signature=None,
            model_assumptions=None,
            evidence=(),
            unmodeled_reason=f"invalid_request:{type(exc).__name__}",
        )
    lower = estimate_ideal_hardware_lower_bound(req.inputs)
    prediction = predict_with_provider(req, provider)
    measured_value = req.measured_device_ms if measured_device_ms is None else measured_device_ms
    measured, measurement_limitation = _coerce_measured(measured_value)
    if lower.status != MODELED:
        return ModelingAlignment(
            status=UNMODELED,
            measured_device_ms=measured,
            ideal_hardware_lower_bound_ms=None,
            model_predicted_ms=None,
            model_provider=None,
            model_provider_version=None,
            adapter_protocol_version=None,
            input_signature=req.input_signature,
            model_assumptions=None,
            evidence=(),
            unmodeled_reason=lower.unmodeled_reason,
            measurement_limitation=measurement_limitation,
        )
    if prediction.status != MODELED:
        return ModelingAlignment(
            status=UNMODELED,
            measured_device_ms=measured,
            ideal_hardware_lower_bound_ms=lower.lower_bound_ms,
            model_predicted_ms=None,
            model_provider=None,
            model_provider_version=None,
            adapter_protocol_version=None,
            input_signature=req.input_signature,
            model_assumptions=None,
            evidence=(),
            unmodeled_reason=prediction.unmodeled_reason,
            measurement_limitation=measurement_limitation,
        )
    return ModelingAlignment(
        status=MODELED,
        measured_device_ms=measured,
        ideal_hardware_lower_bound_ms=lower.lower_bound_ms,
        model_predicted_ms=prediction.predicted_ms,
        model_provider=f"{prediction.provider_name}@{prediction.provider_version}",
        model_provider_version=prediction.provider_version,
        adapter_protocol_version=prediction.adapter_protocol_version,
        input_signature=prediction.input_signature,
        model_assumptions=prediction.model_assumptions,
        evidence=prediction.evidence,
        unmodeled_reason=None,
        measurement_limitation=measurement_limitation,
    )

@dataclass(frozen=True)
class MsmodelingAdapterStatus:
    adapter_protocol_version: str
    installed: bool
    verified: bool
    provider_name: str | None
    provider_version: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_protocol_version": self.adapter_protocol_version,
            "installed": self.installed,
            "verified": self.verified,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "reason": self.reason,
        }

def probe_msmodeling_adapter() -> MsmodelingAdapterStatus:
    """Report install state only; no source-verified adapter is bundled here."""
    try:
        spec = importlib.util.find_spec(MSMODELING_MODULE)
    except Exception as exc:  # noqa: BLE001 - probe must not raise into callers
        return MsmodelingAdapterStatus(
            adapter_protocol_version=ADAPTER_PROTOCOL_VERSION,
            installed=False,
            verified=False,
            provider_name=None,
            provider_version=None,
            reason=f"msmodeling_probe_failed:{type(exc).__name__}",
        )
    if spec is None:
        return MsmodelingAdapterStatus(
            adapter_protocol_version=ADAPTER_PROTOCOL_VERSION,
            installed=False,
            verified=False,
            provider_name=None,
            provider_version=None,
            reason="msmodeling_not_installed",
        )
    return MsmodelingAdapterStatus(
        adapter_protocol_version=ADAPTER_PROTOCOL_VERSION,
        installed=True,
        verified=False,
        provider_name=None,
        provider_version=None,
        reason="msmodeling_provider_unverified",
    )


_HAP_DTYPE_MAP: dict[str, tuple[str, int]] = {
    "bf16": ("bf16", 2), "bfloat16": ("bf16", 2), "dt_bf16": ("bf16", 2),
    "fp16": ("fp16", 2), "float16": ("fp16", 2), "half": ("fp16", 2), "dt_float16": ("fp16", 2),
    "fp32": ("fp32", 4), "float32": ("fp32", 4), "float": ("fp32", 4), "dt_float32": ("fp32", 4), "dt_float": ("fp32", 4),
    "int8": ("int8", 1), "i8": ("int8", 1), "uint8": ("int8", 1), "u8": ("int8", 1), "dt_int8": ("int8", 1),
    "fp8": ("fp8_e4m3", 1), "fp8_e4m3": ("fp8_e4m3", 1), "float8_e4m3fn": ("fp8_e4m3", 1),
    "fp8_e5m2": ("fp8_e5m2", 1), "float8_e5m2": ("fp8_e5m2", 1),
}


class OperatorHapEvaluation(dict[str, Any]):
    """Dict-compatible HAP & Roofline evaluation record supporting attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    as_dict = to_dict


def compute_hap_score(
    t_cand_us: float | None,
    t_baseline_us: float | None,
    t_hw_us: float | None,
) -> float | None:
    """Compute Hardware-Anchored Performance (HAP) from candidate, baseline, and Roofline floor.

    Formula (cann-bench):
      HAP = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))

    Returns ``None`` when any anchor is non-finite or ``<= 0``, when
    ``T_baseline <= T_HW``, or when the denominator is ``<= 0``.
    """
    if not (_finite(t_cand_us) and _finite(t_baseline_us) and _finite(t_hw_us)):
        return None
    cand = float(t_cand_us)
    base = float(t_baseline_us)
    hw = float(t_hw_us)
    if cand <= 0.0 or base <= 0.0 or hw <= 0.0:
        return None
    base_gap = base - hw
    cand_gap = cand - hw
    denom = cand_gap + base_gap
    if base_gap <= 0.0 or denom <= 0.0:
        return None
    return base_gap / denom


def _unmodeled_hap(
    reason: str,
    *,
    status: str = UNMODELED,
    sku_name: str | None = None,
    arch_family: str | None = None,
    dtype_key: str | None = None,
    compute_ops: float | None = None,
    bytes_moved: float | None = None,
    t_cand_us: float | None = None,
    t_baseline_us: float | None = None,
) -> OperatorHapEvaluation:
    return OperatorHapEvaluation({
        "status": status, "unmodeled_reason": reason, "sku": sku_name,
        "arch_family": arch_family, "dtype": dtype_key, "compute_ops": compute_ops,
        "bytes_moved": bytes_moved, "t_cand_us": t_cand_us, "t_baseline_us": t_baseline_us,
        "t_hw_us": None, "t_compute_us": None, "t_cube_us": None, "t_vector_us": None,
        "t_mte_us": None, "hap_score": None, "hap": None, "bound_pipe": None,
        "bottleneck": None, "achieved_mfu": None, "mfu": None, "achieved_cube_mfu": None,
        "achieved_vector_mfu": None, "achieved_hbm_bw_utilization": None,
        "hbm_bw_utilization": None, "roofline_efficiency": None, "speedup_vs_baseline": None,
        "achieved_tflops": None, "achieved_hbm_bw_gbps": None, "peak_compute_tflops": None,
        "peak_hbm_bw_gbps": None, "arithmetic_intensity_ops_per_byte": None,
        "ridge_point_ops_per_byte": None,
    })


def evaluate_operator_hap(
    sku: Any,
    compute_ops: float | None,
    bytes_moved: float | None,
    t_cand_us: float | None,
    t_baseline_us: float | None,
    dtype: str = "bf16",
    *,
    op_pipe: str | None = None,
    cube_ops: float | None = None,
    vector_ops: float | None = None,
    clock_mhz: float | None = None,
) -> OperatorHapEvaluation:
    """Evaluate operator Roofline lower bound T_HW, HAP score, MFU, HBM utilization, and bound pipe."""
    raw_dtype = str(dtype or "").strip().lower().replace("-", "_")
    if raw_dtype not in _HAP_DTYPE_MAP:
        return _unmodeled_hap(f"unsupported_dtype:{dtype}", dtype_key=raw_dtype or None)
    dtype_key, elem_bytes = _HAP_DTYPE_MAP[raw_dtype]

    if not (_finite(compute_ops) and float(compute_ops) >= 0.0):
        return _unmodeled_hap("invalid_compute_ops", dtype_key=dtype_key)
    if not (_finite(bytes_moved) and float(bytes_moved) >= 0.0):
        return _unmodeled_hap("invalid_bytes_moved", dtype_key=dtype_key, compute_ops=float(compute_ops))
    if float(compute_ops) == 0.0 and float(bytes_moved) == 0.0 and not (_finite(cube_ops) or _finite(vector_ops)):
        return _unmodeled_hap("zero_workload", dtype_key=dtype_key, compute_ops=0.0, bytes_moved=0.0)

    cand_f = float(t_cand_us) if _finite(t_cand_us) and float(t_cand_us) > 0.0 else None
    base_f = float(t_baseline_us) if _finite(t_baseline_us) and float(t_baseline_us) > 0.0 else None
    sku_name: str | None = None
    arch_family: str | None = None
    cube_peak_ops_per_s: float | None = None
    vector_peak_ops_per_s: float | None = None
    hbm_peak_bytes_per_s: float | None = None

    if isinstance(sku, HardwareSpec) or (
        isinstance(sku, Mapping) and "peak_flops_per_s" in sku and "memory_bandwidth_bytes_per_s" in sku
    ):
        hw_spec = sku if isinstance(sku, HardwareSpec) else HardwareSpec(**dict(sku))
        errors = hw_spec.validation_errors()
        if errors:
            return _unmodeled_hap("invalid_hardware:" + ",".join(errors), dtype_key=dtype_key)
        sku_name = hw_spec.name
        cube_peak_ops_per_s = vector_peak_ops_per_s = float(hw_spec.peak_flops_per_s)
        hbm_peak_bytes_per_s = float(hw_spec.memory_bandwidth_bytes_per_s)
    else:
        if isinstance(sku, DeviceCapabilityProfile):
            profile = sku
        elif hasattr(sku, "soc_version") and hasattr(sku, "cube_core_cnt"):
            profile = resolve_device_capability_profile(config=sku)
        elif isinstance(sku, Mapping):
            profile = resolve_device_capability_profile(sku_profile=sku)
        elif isinstance(sku, str) and sku.strip():
            token = sku.strip()
            if is_ambiguous_family_token(token):
                return _unmodeled_hap(
                    AMBIGUOUS_SKU_REASON, status="ambiguous_sku", sku_name=token, dtype_key=dtype_key,
                    compute_ops=float(compute_ops), bytes_moved=float(bytes_moved), t_cand_us=cand_f, t_baseline_us=base_f,
                )
            if token not in BUILTIN_SKU_CATALOG:
                return _unmodeled_hap(
                    f"unknown_sku:{token}", sku_name=token, dtype_key=dtype_key,
                    compute_ops=float(compute_ops), bytes_moved=float(bytes_moved), t_cand_us=cand_f, t_baseline_us=base_f,
                )
            profile = resolve_device_capability_profile(soc=token)
        else:
            return _unmodeled_hap("missing_sku", dtype_key=dtype_key)

        if profile.status == "ambiguous_sku" or profile.unmodeled_reason:
            return _unmodeled_hap(
                profile.unmodeled_reason or AMBIGUOUS_SKU_REASON, status=profile.status,
                sku_name=profile.clock_and_topology_spec.soc_version or profile.clock_and_topology_spec.short_soc_version,
                arch_family=profile.clock_and_topology_spec.arch_family, dtype_key=dtype_key,
                compute_ops=float(compute_ops), bytes_moved=float(bytes_moved), t_cand_us=cand_f, t_baseline_us=base_f,
            )

        sku_name = profile.clock_and_topology_spec.soc_version or profile.clock_and_topology_spec.short_soc_version
        arch_family = profile.clock_and_topology_spec.arch_family
        freq_mhz = (
            float(clock_mhz) if _finite(clock_mhz) and float(clock_mhz) > 0
            else (profile.clock_and_topology_spec.runtime_clock_mhz or profile.clock_and_topology_spec.configured_clock_mhz)
        )
        aic_cnt = profile.cube_spec.aic_count_per_die
        aiv_cnt = profile.vector_spec.aiv_count_per_die
        ops_per_cyc = profile.cube_spec.ops_per_cycle_per_aic.get(dtype_key)
        if ops_per_cyc is None:
            return _unmodeled_hap(
                f"unsupported_dtype_for_sku:{dtype_key}@{sku_name}",
                sku_name=sku_name, arch_family=arch_family, dtype_key=dtype_key,
            )
        if aic_cnt and freq_mhz:
            cube_peak_ops_per_s = float(aic_cnt) * float(ops_per_cyc) * float(freq_mhz) * 1e6
        if aiv_cnt and freq_mhz and profile.vector_spec.simd_width_bytes > 0:
            vec_ops_per_cyc = float(profile.vector_spec.simd_width_bytes) / float(elem_bytes)
            vector_peak_ops_per_s = float(aiv_cnt) * vec_ops_per_cyc * float(freq_mhz) * 1e6
        if profile.memory_hierarchy_spec.hbm_peak_bw_gbps:
            hbm_peak_bytes_per_s = float(profile.memory_hierarchy_spec.hbm_peak_bw_gbps) * 1e9

    if not (cube_peak_ops_per_s and hbm_peak_bytes_per_s):
        return _unmodeled_hap("missing_hardware_peaks", sku_name=sku_name, arch_family=arch_family, dtype_key=dtype_key)

    is_vec_pipe = (op_pipe or "").strip().lower() in {"vector", "vec", "aiv", "vector_bound"}
    eff_cube_ops = float(cube_ops) if _finite(cube_ops) and float(cube_ops) >= 0.0 else (0.0 if is_vec_pipe else float(compute_ops))
    eff_vec_ops = float(vector_ops) if _finite(vector_ops) and float(vector_ops) >= 0.0 else (float(compute_ops) if is_vec_pipe else 0.0)
    total_ops = eff_cube_ops + eff_vec_ops
    bytes_f = float(bytes_moved)

    t_cube_us = (eff_cube_ops / cube_peak_ops_per_s) * 1e6 if eff_cube_ops > 0.0 else 0.0
    t_vector_us = (eff_vec_ops / vector_peak_ops_per_s) * 1e6 if (eff_vec_ops > 0.0 and vector_peak_ops_per_s) else 0.0
    t_compute_us = max(t_cube_us, t_vector_us)
    t_mte_us = (bytes_f / hbm_peak_bytes_per_s) * 1e6 if bytes_f > 0.0 else 0.0
    t_hw_us = max(t_cube_us, t_vector_us, t_mte_us)

    if t_mte_us >= t_cube_us and t_mte_us >= t_vector_us and (t_mte_us > 0.0 or t_compute_us == 0.0):
        bound_pipe = "mte_bound"
    elif t_vector_us > t_cube_us and t_vector_us > t_mte_us:
        bound_pipe = "vector_bound"
    else:
        bound_pipe = "cube_bound"

    hap_score = compute_hap_score(cand_f, base_f, t_hw_us)
    active_peak_ops = (
        vector_peak_ops_per_s
        if (bound_pipe == "vector_bound" or is_vec_pipe or (eff_vec_ops > eff_cube_ops and vector_peak_ops_per_s))
        else cube_peak_ops_per_s
    )
    achieved_mfu = (t_compute_us / cand_f) if cand_f is not None else None
    achieved_cube_mfu = (t_cube_us / cand_f) if cand_f is not None else None
    achieved_vector_mfu = (t_vector_us / cand_f) if cand_f is not None else None
    achieved_hbm_util = (t_mte_us / cand_f) if cand_f is not None else None
    roofline_eff = (t_hw_us / cand_f) if cand_f is not None else None
    speedup = (base_f / cand_f) if (cand_f is not None and base_f is not None) else None
    achieved_tflops = (total_ops / (cand_f * 1e-6)) / 1e12 if cand_f is not None else None
    achieved_bw_gbps = (bytes_f / (cand_f * 1e-6)) / 1e9 if cand_f is not None else None

    return OperatorHapEvaluation({
        "status": MODELED, "unmodeled_reason": None, "sku": sku_name,
        "arch_family": arch_family, "dtype": dtype_key, "compute_ops": total_ops,
        "bytes_moved": bytes_f, "t_cand_us": cand_f, "t_baseline_us": base_f,
        "t_hw_us": t_hw_us, "t_compute_us": t_compute_us, "t_cube_us": t_cube_us,
        "t_vector_us": t_vector_us, "t_mte_us": t_mte_us, "hap_score": hap_score,
        "hap": hap_score, "bound_pipe": bound_pipe, "bottleneck": bound_pipe,
        "achieved_mfu": achieved_mfu, "mfu": achieved_mfu,
        "achieved_cube_mfu": achieved_cube_mfu, "achieved_vector_mfu": achieved_vector_mfu,
        "achieved_hbm_bw_utilization": achieved_hbm_util, "hbm_bw_utilization": achieved_hbm_util,
        "roofline_efficiency": roofline_eff, "speedup_vs_baseline": speedup,
        "achieved_tflops": achieved_tflops, "achieved_hbm_bw_gbps": achieved_bw_gbps,
        "peak_compute_tflops": active_peak_ops / 1e12 if active_peak_ops else None,
        "peak_hbm_bw_gbps": hbm_peak_bytes_per_s / 1e9,
        "arithmetic_intensity_ops_per_byte": (total_ops / bytes_f) if bytes_f > 0.0 else None,
        "ridge_point_ops_per_byte": (active_peak_ops / hbm_peak_bytes_per_s) if active_peak_ops else None,
    })


model_kernel = predict_with_provider
msmodeling_adapter_status = probe_msmodeling_adapter
align = align_modeling
__all__ = [
    "ADAPTER_PROTOCOL_VERSION", "IDEAL_BOUND_FORMULA_VERSION", "MODELED", "UNMODELED",
    "HardwareSpec", "KernelInputs", "IdealLowerBound", "ProviderIdentity",
    "ProviderUnmodeledError", "ProviderPrediction", "ModelingRequest", "ModelingProvider",
    "ModelPrediction", "ModelingAlignment", "MsmodelingAdapterStatus",
    "OperatorHapEvaluation", "compute_hap_score", "evaluate_operator_hap",
    "estimate_ideal_hardware_lower_bound", "ideal_hardware_lower_bound",
    "predict_with_provider", "model_kernel", "align_modeling", "align",
    "probe_msmodeling_adapter", "msmodeling_adapter_status",
]
