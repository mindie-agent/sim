"""Observed resource-service rate extraction, selection and binding.

[Role] Select and bind observed resource-service rates for hardware profiles.
[Invariants] Explicit provenance and positive physical rates.
[Timeouts & Locks] Pure in-memory lookup; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence


_RATE_SCOPES = frozenset({"aggregate", "per_owner"})


def _text(value, name, *, default=None):
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"resource_service_requires_nonempty_{name}")
    return value.strip()


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resource_service_{name}_requires_number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"resource_service_{name}_requires_finite")
    if positive and result <= 0:
        raise ValueError(f"resource_service_{name}_requires_positive")
    if result < 0:
        raise ValueError(f"resource_service_{name}_requires_nonnegative")
    return result


def resolve_nominal_rate(resource, nominal_rates, *, work_unit, hardware=None,
                         rate_scope="aggregate"):
    """Return one hardware-nominal rate, requiring its actual unit and scope.

    A bare number is rejected: it would silently reinterpret ``device.rates``
    in the caller's ``work_unit``/``rate_scope``.  A typed mapping must declare
    ``rate``, the hardware's actual ``work_unit`` and ``rate_scope``.  A unit
    conversion is accepted only when ``conversion`` explicitly declares the
    target unit and the source-units-per-target-unit factor; no implicit
    bytes-to-rows or aggregate-to-per-owner conversion is made.
    """
    if nominal_rates is None:
        return None
    if not isinstance(nominal_rates, Mapping):
        raise ValueError("resource_service_nominal_rates_requires_mapping")
    raw = nominal_rates.get(resource)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(
            "resource_service_bare_nominal_rate_requires_explicit_unit_and_scope")
    unknown = set(raw) - {
        "rate", "work_unit", "rate_scope", "hardware", "conversion", "source"}
    if unknown:
        raise ValueError("resource_service_nominal_rate_unknown_fields:"
                         + ",".join(sorted(str(item) for item in unknown)))
    if "work_unit" not in raw:
        raise ValueError(
            "resource_service_nominal_rate_requires_explicit_work_unit")
    if "rate_scope" not in raw:
        raise ValueError(
            "resource_service_nominal_rate_requires_explicit_rate_scope")
    source_unit = _text(raw.get("work_unit"), "nominal_rate_work_unit")
    declared_scope = _text(raw.get("rate_scope"), "nominal_rate_scope")
    if declared_scope not in _RATE_SCOPES:
        raise ValueError("resource_service_nominal_rate_scope_invalid")
    if declared_scope != rate_scope:
        raise ValueError("resource_service_nominal_rate_scope_mismatch")
    declared_hardware = raw.get("hardware")
    if hardware and declared_hardware and str(declared_hardware) != str(hardware):
        raise ValueError("resource_service_nominal_rate_hardware_mismatch")
    rate = _number(raw.get("rate"), "nominal_rate", positive=True)
    conversion = raw.get("conversion")
    if source_unit == work_unit:
        if conversion is not None:
            raise ValueError(
                "resource_service_nominal_rate_conversion_not_needed")
        return rate
    if not isinstance(conversion, Mapping):
        raise ValueError("resource_service_nominal_rate_unit_mismatch")
    unknown = set(conversion) - {
        "target_work_unit", "source_units_per_target_unit"}
    if unknown:
        raise ValueError("resource_service_nominal_rate_conversion_unknown_fields:"
                         + ",".join(sorted(str(item) for item in unknown)))
    target_unit = _text(conversion.get("target_work_unit"),
                        "conversion_target_work_unit")
    if target_unit != work_unit:
        raise ValueError("resource_service_nominal_rate_unit_mismatch")
    factor = _number(conversion.get("source_units_per_target_unit"),
                     "conversion_factor", positive=True)
    return rate / factor


from typing import Any, Mapping, Sequence


_SPARSE_OWNER_COUNT = 32
_SPARSE_GROUP_SIZE = 64
_SPARSE_MAX_SINGLE_CORE_HEADS = 64
_SPARSE_MAX_SPLIT_HEADS = 128
_SPARSE_QUERY_WIDTH = 576
_SPARSE_VALUE_WIDTH = 512
_SPARSE_KV_TILE_ROWS = 128
_SPARSE_KV_ROW_ALIGNMENT = 16


def _align_sparse_fractal(value: int) -> int:
    return ((value + _SPARSE_KV_ROW_ALIGNMENT - 1)
            // _SPARSE_KV_ROW_ALIGNMENT) * _SPARSE_KV_ROW_ALIGNMENT


def sparse_mte1_bytes(q_len: int, context: int, kv_len: int, *, causal: bool = True,
                      owners: int = _SPARSE_OWNER_COUNT,
                      g: int = _SPARSE_GROUP_SIZE, dq: int = _SPARSE_QUERY_WIDTH,
                      dv: int = _SPARSE_VALUE_WIDTH,
                      s2_tile: int = _SPARSE_KV_TILE_ROWS) -> dict[str, Any]:
    """Derive per-critical-owner QK/PV L1->L0A/B bytes for one geometry."""
    q_len, context, kv_len, owners = int(q_len), int(context), int(kv_len), int(owners)
    g, dq, dv, s2_tile = int(g), int(dq), int(dv), int(s2_tile)
    if (q_len <= 0 or owners <= 0 or kv_len <= 0 or context < 0
            or g <= 0 or g > _SPARSE_MAX_SPLIT_HEADS
            or dq <= 0 or dv <= 0 or s2_tile <= 0):
        raise ValueError("sparse_mte1_geometry_invalid")
    split_g = g > _SPARSE_MAX_SINGLE_CORE_HEADS
    if split_g:
        query_owners = owners // 2
        if query_owners <= 0:
            raise ValueError("sparse_mte1_geometry_invalid")
        head_splits = ((g + 1) // 2, g // 2)
    else:
        query_owners = owners
        head_splits = (g,)
    aligned_dq = _align_sparse_fractal(dq)
    aligned_dv = _align_sparse_fractal(dv)
    width = -(-q_len // query_owners)
    per_owner = []
    for start in range(0, q_len, width):
        stop = min(start + width, q_len)
        for local_g in head_splits:
            aligned_m = _align_sparse_fractal(local_g)
            qk = pv = ub = tiles = 0
            for index in range(start, stop):
                selected = min(kv_len, context + index + 1) if causal else min(
                    kv_len, context + q_len)
                remaining = selected
                while remaining > 0:
                    rows = min(s2_tile, remaining)
                    aligned_kv = _align_sparse_fractal(rows)
                    qk += 2 * aligned_m * aligned_dq + 2 * aligned_kv * aligned_dq
                    pv += 2 * aligned_m * aligned_kv + 2 * aligned_kv * aligned_dv
                    ub += 2 * local_g * s2_tile
                    tiles += 1
                    remaining -= rows
            per_owner.append({"qk_pv_bytes": qk + pv, "qk_bytes": qk, "pv_bytes": pv,
                              "ub_bytes": ub, "tiles": tiles})
    critical_index = max(range(len(per_owner)),
                         key=lambda index: per_owner[index]["qk_pv_bytes"])
    critical = per_owner[critical_index]
    return {
        "q_len": q_len, "context": context, "kv_len": kv_len, "causal": bool(causal),
        "owners": owners, "g": g, "dq": dq, "dv": dv,
        "s2_tile": s2_tile, "per_owner_width": width,
        "critical_owner_index": critical_index,
        "critical_qk_pv_bytes": critical["qk_pv_bytes"],
        "critical_qk_bytes": critical["qk_bytes"],
        "critical_pv_bytes": critical["pv_bytes"],
        "critical_ub_bytes": critical["ub_bytes"],
        "critical_tiles": critical["tiles"],
        "total_qk_pv_bytes": sum(item["qk_pv_bytes"] for item in per_owner),
        "total_ub_bytes": sum(item["ub_bytes"] for item in per_owner),
        "total_tiles": sum(item["tiles"] for item in per_owner),
    }


def _digest(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def identity_digest(value: Any) -> str:
    """Normalize one declared image/source digest exactly once."""
    return _digest(value)


def c7_applicability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the original C7 raw-record applicability contract.

    The raw selector, normalized service and direct binding all consume these
    conditions.  Source identity, native implementation, run state, owner
    count and the measured-q work domain stay executable; they are not merely
    provenance.
    """
    if not isinstance(record, Mapping):
        raise ValueError("c7_applicability_record_requires_mapping")
    source = dict(record.get("source_identity") or {})
    measured_q = int(record.get("measured_q_tokens") or 0)
    conditions = {
        "hardware": str(record.get("hardware") or ""),
        "dtype": str(record.get("dtype") or "").lower(),
        "native_type": str(record.get("native_type") or ""),
        "phase": str(record.get("phase") or "").lower(),
        "op_state": str(record.get("op_state") or "").lower(),
        "owners": int(record.get("owners") or 0),
        "source_identity.source_scope": str(source.get("source_scope") or ""),
        "source_identity.image_digest": identity_digest(
            source.get("image_digest")),
        "geometry.q_len": {"min": measured_q},
    }
    optional = {}
    if record.get("mix_block_num") is not None:
        optional["mix_block_num"] = int(record.get("mix_block_num"))
    geometry = dict(record.get("geometry_binding") or {})
    return {
        "conditions": conditions,
        "optional_conditions": optional,
        "work_domain": {
            "measured_q_tokens": measured_q,
            "q_len_min": measured_q,
            "geometry_binding": geometry,
            "measured_critical_bytes": int(
                record.get("measured_critical_bytes") or 0),
            "owners": int(record.get("owners") or 0),
            "work_unit": "bytes",
            "rate_scope": "per_owner",
            "rate_bytes_per_s": float(record.get("rate_bytes_per_s") or 0.0),
        },
    }


_CONDITION_MISSING = object()
_SOURCE_SHA_PREFIX_LEN = 12


def _profiler_service_id(record: Mapping[str, Any], resource: str) -> str:
    source_sha = str(record.get("source_sha256") or "")
    return (
        f"profiler:{resource}:"
        f"{record.get('shape_class') or 'default'}:"
        f"{source_sha[:_SOURCE_SHA_PREFIX_LEN] or 'unhashed'}"
    )


def _service_id(service: Any) -> str:
    service_id = getattr(service, "service_id", None)
    if service_id is not None:
        return str(service_id)
    if isinstance(service, Mapping):
        return str(service.get("service_id") or service.get("id") or "")
    return ""


def _condition_value(context, path):
    value = context
    for part in str(path).split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return _CONDITION_MISSING
            value = value[part]
        elif (isinstance(value, (list, tuple)) and part.isdigit()
              and int(part) < len(value)):
            value = value[int(part)]
        else:
            return _CONDITION_MISSING
    return value


def _condition_matches(actual, expected, path=""):
    if str(path).endswith("image_digest"):
        return identity_digest(actual) == identity_digest(expected)
    if str(path) == "dtype":
        return str(actual).lower() == str(expected).lower()
    if isinstance(expected, Mapping) and set(expected) <= {"min", "max"}:
        try:
            number = float(actual)
            if "min" in expected and number < float(expected["min"]):
                return False
            if "max" in expected and number > float(expected["max"]):
                return False
            return True
        except (TypeError, ValueError):
            return False
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)):
        return math.isclose(float(actual), float(expected),
                            rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return (actual.keys() == expected.keys()
                and all(_condition_matches(actual[key], expected[key], path)
                        for key in expected))
    if (isinstance(actual, (list, tuple))
            and isinstance(expected, (list, tuple))):
        return (len(actual) == len(expected)
                and all(_condition_matches(a, b, path)
                        for a, b in zip(actual, expected)))
    return actual == expected


def service_conditions_match(service, context):
    """Check only the conditions the service actually declares.

    Extra keys in a broader physical contract (unrelated rates, modes, owners or
    environment fields) are not identity.  A declared rate/mode/unit/environment
    condition must still match, so a real change in the depended-on supply is
    rejected.  ``conditions`` may use dotted paths such as
    ``resource_units.mte2_aiv`` or ``rates.hbm``.
    """
    if not isinstance(context, Mapping):
        raise ValueError("service_condition_context_requires_mapping")
    conditions = getattr(service, "conditions", None)
    optional = getattr(service, "optional_conditions", None)
    if isinstance(service, Mapping):
        if conditions is None and "conditions" in service:
            conditions = service["conditions"]
        if optional is None and "optional_conditions" in service:
            optional = service["optional_conditions"]
    if conditions is None:
        if optional is None:
            raise ValueError("service_condition_service_requires_conditions")
        conditions = {}
    if not isinstance(conditions, Mapping):
        raise ValueError("service_condition_conditions_requires_mapping")
    checks = []
    for path, expected in conditions.items():
        actual = _condition_value(context, path)
        present = actual is not _CONDITION_MISSING
        matched = present and _condition_matches(actual, expected, path)
        checks.append({
            "path": str(path),
            "expected": deepcopy(expected),
            "actual": deepcopy(actual) if present else None,
            "present": present,
            "matched": matched,
            "optional": False,
        })
    if optional:
        if not isinstance(optional, Mapping):
            raise ValueError("service_optional_conditions_requires_mapping")
        for path, expected in optional.items():
            actual = _condition_value(context, path)
            present = actual is not _CONDITION_MISSING
            if not present or actual is None:
                checks.append({
                    "path": str(path), "expected": deepcopy(expected),
                    "actual": None, "present": present, "matched": True,
                    "optional": True,
                })
                continue
            matched = _condition_matches(actual, expected, path)
            checks.append({
                "path": str(path), "expected": deepcopy(expected),
                "actual": deepcopy(actual), "present": True,
                "matched": matched, "optional": True,
            })
    return {
        "applies": all(check["matched"] for check in checks),
        "checks": checks,
        "policy": ("only_declared_condition_paths; unrelated physical "
                   "contract keys are ignored"),
    }


def _condition_failure_reason(service, result):
    failed = [check for check in result["checks"] if not check["matched"]]
    if not failed:
        return "conditions_matched"
    parts = []
    for check in failed:
        actual = check["actual"] if check["present"] else "<missing>"
        parts.append(f"{check['path']}={actual!r}!={check['expected']!r}")
    return _service_id(service) + ":" + ";".join(parts)


def select_resource_service_candidate(services, context, *, service_id=None):
    """Narrow by optional ID, then apply the one condition matcher to all.

    Explicit ID, single candidate and automatic selection all pass through the
    same condition check.  A missing ID, missing condition, value mismatch or
    ambiguity raises a specific error; there is no nominal fallback.
    """
    if not isinstance(context, Mapping):
        raise ValueError("resource_service_context_requires_mapping")
    candidates = list(services or ())
    if service_id is not None:
        candidates = [item for item in candidates
                      if _service_id(item) == service_id]
        if not candidates:
            raise ValueError("resource_service_id_not_found:" + str(service_id))
    checked = []
    selected = []
    for service in candidates:
        result = service_conditions_match(service, context)
        checked.append((service, result))
        if result["applies"]:
            selected.append(service)
    if len(selected) == 1:
        return selected[0]
    if not selected:
        reasons = ";".join(
            _condition_failure_reason(service, result)
            for service, result in checked)
        raise ValueError("resource_service_conditions_have_no_match:" + reasons)
    raise ValueError("resource_service_conditions_are_ambiguous:"
                     + ",".join(sorted(_service_id(item) for item in selected)))


def c7_applicability_matches(record: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """Apply the same extracted conditions used by normalized selection."""
    return bool(service_conditions_match(c7_applicability(record), context)["applies"])


def select_resource_service(
    resource_services: Sequence[Mapping[str, Any]] | None, *,
    native_type: str = "KvQuantSparseFlashAttention",
    hardware: str, source_identity: Mapping[str, Any] | None,
    dtype: str = "bf16", q_len: int, context: int, kv_len: int,
    causal: bool = True, owners: int = _SPARSE_OWNER_COUNT, phase: str = "prefill",
    op_state: str = "dynamic", mix_block_num: int | None = None,
) -> dict[str, Any] | None:
    """Return the measured MTE1 service for one saturated query geometry."""
    q_len, owners = int(q_len), int(owners)
    applicability_context = {
        "hardware": str(hardware),
        "dtype": str(dtype),
        "native_type": str(native_type),
        "phase": str(phase),
        "op_state": str(op_state),
        "owners": owners,
        "source_identity": dict(source_identity or {}),
        "geometry": {"q_len": q_len},
        "mix_block_num": mix_block_num,
    }
    candidates = []
    for record in resource_services or ():
        if not isinstance(record, Mapping):
            continue
        if str(record.get("resource")) != "mte1":
            continue
        if record.get("rate_bytes_per_s") is None:
            continue
        applicability = c7_applicability(record)
        candidates.append({
            "service_id": _profiler_service_id(record, "mte1"),
            "conditions": applicability["conditions"],
            "optional_conditions": applicability["optional_conditions"],
            "record": record,
        })
    if not candidates:
        return None
    try:
        selected = select_resource_service_candidate(
            candidates, applicability_context)
    except ValueError as exc:
        if str(exc).startswith("resource_service_conditions_have_no_match:"):
            return None
        raise
    record = selected["record"]
    try:
        geometry = sparse_mte1_bytes(
            q_len, int(context), int(kv_len), causal=bool(causal), owners=owners,
            g=int((record.get("geometry_binding") or {}).get("g", _SPARSE_GROUP_SIZE)),
            dq=int((record.get("geometry_binding") or {}).get("dq", _SPARSE_QUERY_WIDTH)),
            dv=int((record.get("geometry_binding") or {}).get("dv", _SPARSE_VALUE_WIDTH)),
            s2_tile=int((record.get("geometry_binding") or {}).get("s2_tile", _SPARSE_KV_TILE_ROWS)),
        )
        rate = float(record["rate_bytes_per_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    critical = int(geometry["critical_qk_pv_bytes"])
    return {
        "schema_version": str(record.get("schema_version") or ""),
        "resource": "mte1",
        "native_type": native_type,
        "source_identity": dict(source_identity or {}),
        "hardware": str(hardware),
        "dtype": str(dtype),
        "layout": str(record.get("layout") or ""),
        "phase": str(phase),
        "op_state": str(op_state),
        "shape_class": str(record.get("shape_class") or ""),
        "owners": owners,
        "q_tokens": q_len,
        "context": int(context),
        "kv_len": int(kv_len),
        "causal": bool(causal),
        "measured_q_tokens": int(record.get("measured_q_tokens") or 0),
        "mix_block_num": record.get("mix_block_num"),
        "measured_critical_bytes": int(record.get("measured_critical_bytes") or 0),
        "query_critical_bytes": critical,
        "query_total_bytes": int(geometry["total_qk_pv_bytes"]),
        "rate_bytes_per_s": rate,
        "time_us": critical / rate * 1e6,
        "event_count": int(record.get("event_count") or 0),
        "median_mte1_us": float(record.get("median_mte1_us") or 0.0),
        "geometry_binding": dict(record.get("geometry_binding") or {}),
    }


from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence



RESOURCE_SERVICE_SCHEMA = "mindie-sim.resource-service/1"
RESOURCE_SERVICE_BINDING_SCHEMA = "mindie-sim.resource-service-binding/1"
RESOURCE_SERVICE_PROFILE_SCHEMA = "mindie-sim.resource-service-profile/1"

_SERVICE_SCOPES = frozenset({"independent_resource", "joint_phase"})
_SERVICE_BOUNDARIES = frozenset({"resource_phase", "device_kernel"})
_SUPPLY_KINDS = frozenset({"absolute_rate", "nominal_multiplier", "nominal"})


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"resource_service_{name}_requires_integer")
    if value < minimum:
        raise ValueError(f"resource_service_{name}_requires_at_least_{minimum}")
    return value


def _plain_mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"resource_service_{name}_requires_mapping")
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"resource_service_{name}_requires_json_values") from exc


def _resource_list(record):
    raw = record.get("resources")
    if raw is None:
        raw = [record["resource"]] if record.get("resource") else None
    elif isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("resource_service_requires_a_resource_list")
    resources = tuple(_text(item, "resource") for item in raw)
    if len(set(resources)) != len(resources):
        raise ValueError("resource_service_requires_distinct_resources")
    return resources


@dataclass(frozen=True)
class NormalizedResourceService:
    """One resource or joint-phase service with a single resolved rate.

    ``effective_rate`` is expressed in ``work_unit`` per second in the declared
    ``rate_scope``.  ``aggregate_rate`` and ``per_owner_rate`` are diagnostics
    for callers that hold the other work scope; neither is a second parameter.
    """

    service_id: str
    kind: str
    scope: str
    resources: tuple[str, ...]
    phase: str
    work_unit: str
    boundary: str
    startup_us: float
    response_us: float
    fixed_us: float
    effective_rate: float
    rate_scope: str
    active_units: int
    aggregate_rate: float
    per_owner_rate: float
    nominal_rate: float | None = None
    nominal_multiplier: float | None = None
    conditions: Mapping = field(default_factory=dict)
    optional_conditions: Mapping = field(default_factory=dict)
    identification: Mapping = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)
    rate_representation: Mapping = field(default_factory=dict)
    normalization: Mapping = field(default_factory=dict)
    model_id: str = field(init=False)

    def __post_init__(self):
        if self.kind not in {"resource_rate", "joint_phase"}:
            raise ValueError("resource_service_kind_unknown")
        if self.scope not in _SERVICE_SCOPES:
            raise ValueError("resource_service_scope_unknown")
        if self.scope == "independent_resource":
            if self.kind != "resource_rate" or len(self.resources) != 1:
                raise ValueError("resource_service_independent_scope_requires_one_resource_rate")
        elif self.kind != "joint_phase":
            raise ValueError("resource_service_joint_scope_requires_joint_phase_kind")
        if not self.resources or any(not isinstance(item, str) or not item
                                     for item in self.resources):
            raise ValueError("resource_service_requires_nonempty_resources")
        if self.active_units < 1:
            raise ValueError("resource_service_requires_positive_active_units")
        if not self.service_id or not self.phase or not self.work_unit:
            raise ValueError("resource_service_requires_id_phase_and_work_unit")
        if self.boundary not in _SERVICE_BOUNDARIES:
            raise ValueError("resource_service_boundary_must_be_resource_phase_or_device_kernel")
        if self.rate_scope not in _RATE_SCOPES:
            raise ValueError("resource_service_rate_scope_unknown")
        if not math.isclose(
                self.fixed_us, self.startup_us + self.response_us,
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("resource_service_fixed_must_equal_startup_plus_response")
        if self.effective_rate <= 0 or not math.isfinite(self.effective_rate):
            raise ValueError("resource_service_effective_rate_requires_positive_finite")
        if self.rate_scope == "aggregate":
            if not math.isclose(self.aggregate_rate, self.effective_rate,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("resource_service_aggregate_rate_mismatch")
            if not math.isclose(
                    self.per_owner_rate, self.effective_rate / self.active_units,
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("resource_service_per_owner_rate_mismatch")
        else:
            if not math.isclose(self.per_owner_rate, self.effective_rate,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("resource_service_per_owner_rate_mismatch")
            if not math.isclose(
                    self.aggregate_rate, self.effective_rate * self.active_units,
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("resource_service_aggregate_rate_mismatch")
        if self.nominal_rate is not None:
            _number(self.nominal_rate, "nominal_rate", positive=True)
        if self.nominal_multiplier is not None:
            _number(self.nominal_multiplier, "nominal_multiplier", positive=True)
        if self.nominal_rate is not None and self.nominal_multiplier is not None:
            if not math.isclose(
                    self.nominal_rate * self.nominal_multiplier,
                    self.effective_rate, rel_tol=1e-9, abs_tol=0.0):
                raise ValueError("resource_service_normalized_rate_inconsistent")
        identification = _plain_mapping(self.identification, "identification")
        rate_representation = _plain_mapping(
            self.rate_representation, "rate_representation")
        resource_identified = identification.get("resource_identified", False)
        if not isinstance(resource_identified, bool):
            raise ValueError("resource_service_resource_identified_requires_boolean")
        if resource_identified and self.scope != "independent_resource":
            raise ValueError("joint_service_cannot_identify_an_internal_resource")
        identification["resource_identified"] = resource_identified
        status = _text(identification.get("status", "declared"),
                       "identification_status")
        if status == "identified" and not resource_identified:
            raise ValueError("resource_service_identified_status_requires_identified_resource")
        if (status == "identified"
                and rate_representation.get("input_kind") == "nominal"):
            raise ValueError(
                "resource_service_nominal_rate_cannot_be_identified_effective_service")
        if status == "joint_identified" and self.scope != "joint_phase":
            raise ValueError("resource_service_joint_status_requires_joint_scope")
        if status == "nominal_specification" and \
                rate_representation.get("input_kind") not in (None, "nominal"):
            raise ValueError("resource_service_nominal_status_requires_nominal_supply")
        identification["status"] = status
        object.__setattr__(self, "conditions",
                           _plain_mapping(self.conditions, "conditions"))
        object.__setattr__(self, "optional_conditions",
                           _plain_mapping(self.optional_conditions,
                                          "optional_conditions"))
        object.__setattr__(self, "identification", identification)
        object.__setattr__(self, "provenance", _plain_mapping(self.provenance, "provenance"))
        object.__setattr__(self, "rate_representation", rate_representation)
        normalization = _plain_mapping(self.normalization, "normalization")
        normalization["applied_once"] = True
        object.__setattr__(self, "normalization", normalization)
        payload = {
            "schema": RESOURCE_SERVICE_SCHEMA,
            "kind": self.kind,
            "scope": self.scope,
            "resources": list(self.resources),
            "phase": self.phase,
            "work_unit": self.work_unit,
            "boundary": self.boundary,
            "startup_us": self.startup_us,
            "response_us": self.response_us,
            "fixed_us": self.fixed_us,
            "effective_rate": self.effective_rate,
            "rate_scope": self.rate_scope,
            "active_units": self.active_units,
            "aggregate_rate": self.aggregate_rate,
            "per_owner_rate": self.per_owner_rate,
            "nominal_rate": self.nominal_rate,
            "nominal_multiplier": self.nominal_multiplier,
            "conditions": self.conditions,
            "optional_conditions": self.optional_conditions,
            "identification": {
                key: self.identification.get(key)
                for key in ("status", "resource_identified", "train_work",
                            "heldout_work", "rate_source",
                            "startup_response_separable")
            },
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        object.__setattr__(self, "model_id", hashlib.sha256(text.encode()).hexdigest())

    @property
    def resource(self):
        return self.resources[0] if self.scope == "independent_resource" else None

    @property
    def formula(self):
        if self.fixed_us:
            return (f"T_us = {self.fixed_us:.12g} + work * 1e6 / "
                    f"{self.effective_rate:.12g}")
        return f"T_us = work * 1e6 / {self.effective_rate:.12g}"

    def duration_us(self, work, *, work_scope=None):
        """Evaluate the formula once; ``work`` uses the declared scope by default."""
        work = _number(work, "work")
        selected = self.rate_scope if work_scope is None else str(work_scope)
        if selected == self.rate_scope:
            rate = self.effective_rate
        elif selected == "aggregate":
            rate = self.aggregate_rate
        elif selected == "per_owner":
            rate = self.per_owner_rate
        else:
            raise ValueError("resource_service_work_scope_unknown")
        duration = self.fixed_us + work * 1e6 / rate
        if not math.isfinite(duration):
            raise ValueError("resource_service_duration_requires_finite_result")
        return duration

    def prediction(self, work, *, work_scope=None):
        """Return a normal formula prediction, including work outside training."""
        work = _number(work, "work")
        train = self.identification.get("train_work") or []
        inside = (min(train) <= work <= max(train)) if train else None
        declared = self.duration_us(work, work_scope=work_scope)
        return {
            "id": self.service_id,
            "kind": self.kind,
            "resource": self.resource,
            "resources": list(self.resources),
            "phase": self.phase,
            "work": work,
            "work_unit": self.work_unit,
            "work_scope": work_scope or self.rate_scope,
            "duration_us": declared,
            "duration_us_aggregate": self.duration_us(work, work_scope="aggregate"),
            "duration_us_per_owner": self.duration_us(work, work_scope="per_owner"),
            "fixed_us": self.fixed_us,
            "service_us": declared - self.fixed_us,
            "effective_rate": self.effective_rate,
            "rate_scope": self.rate_scope,
            "basis": "formula",
            "inside_training_range": inside,
            "extrapolated": None if inside is None else not inside,
            "formula": self.formula,
            "model_id": self.model_id,
        }

    def _rate_intervention(self, work, new_effective, *, basis, assumption,
                           nominal_rate=None, multiplier_source=None):
        if not math.isfinite(new_effective) or new_effective <= 0:
            raise ValueError("resource_service_counterfactual_rate_not_finite")
        if self.rate_scope == "aggregate":
            new_aggregate = new_effective
            new_per_owner = new_effective / self.active_units
        else:
            new_aggregate = new_effective * self.active_units
            new_per_owner = new_effective
        work = _number(work, "work")
        durations = [self.fixed_us + work * 1e6 / rate
                     for rate in (new_effective, new_aggregate, new_per_owner)]
        if any(not math.isfinite(value) for value in durations):
            raise ValueError("resource_service_counterfactual_duration_not_finite")
        return {
            "id": self.service_id,
            "kind": self.kind,
            "resource": self.resource,
            "resource_identified": bool(
                self.identification.get("resource_identified")),
            "basis": basis,
            "multiplier_source": multiplier_source,
            "assumption": assumption,
            "work": work,
            "work_unit": self.work_unit,
            "rate_scope": self.rate_scope,
            "nominal_rate": nominal_rate,
            "effective_rate": new_effective,
            "aggregate_rate": new_aggregate,
            "per_owner_rate": new_per_owner,
            "fixed_us": self.fixed_us,
            "duration_us": durations[0],
            "duration_us_aggregate": durations[1],
            "duration_us_per_owner": durations[2],
        }

    def counterfactual(self, work, *, nominal_rate):
        """Scale by a nominal rate, using the declared or inferred multiplier.

        This is a nominal-scaling intervention, not a pure-rate replacement.
        An absolute effective rate with a nominal reference has an inferred
        multiplier (0.75 in the H5 example), so scaling the nominal to the
        effective value keeps the service at the old effective rate.  Use
        ``counterfactual_effective_rate`` for a pure-rate intervention.
        """
        if self.kind != "resource_rate":
            raise ValueError(
                "joint_service_does_not_identify_internal_resource_counterfactual")
        new_nominal = _number(nominal_rate, "counterfactual_nominal_rate",
                              positive=True)
        input_kind = self.rate_representation.get("input_kind")
        if input_kind == "nominal_multiplier":
            if self.nominal_multiplier is None:
                raise ValueError(
                    "resource_service_counterfactual_requires_nominal_rate")
            new_effective = new_nominal * self.nominal_multiplier
            basis = "declared_nominal_multiplier"
        elif input_kind == "nominal":
            if self.nominal_rate is None:
                raise ValueError(
                    "resource_service_counterfactual_requires_nominal_rate")
            new_effective = new_nominal
            basis = "declared_nominal_rate_ratio"
        elif input_kind == "absolute_rate" and self.nominal_rate is not None:
            new_effective = self.effective_rate * new_nominal / self.nominal_rate
            basis = "effective_rate_ratio_to_declared_nominal"
        elif self.nominal_multiplier is not None:
            new_effective = new_nominal * self.nominal_multiplier
            basis = "inferred_nominal_multiplier"
        elif self.nominal_rate is not None:
            new_effective = self.effective_rate * new_nominal / self.nominal_rate
            basis = "effective_rate_ratio_to_declared_nominal"
        else:
            raise ValueError("resource_service_counterfactual_requires_nominal_rate")
        return self._rate_intervention(
            work, new_effective, basis=basis,
            assumption=("nominal scaling changes steady-state rate through the "
                        "declared or inferred multiplier; startup/response stay "
                        "at the observed boundary"),
            nominal_rate=new_nominal,
            multiplier_source=self.rate_representation.get("multiplier_source"))

    def counterfactual_effective_rate(self, work, *, effective_rate):
        """Set the steady-state service rate directly, keeping fixed_us.

        This is the general pure-rate intervention interface for T17/H5.  A
        joint-phase service still cannot be split into an internal resource
        intervention.
        """
        if self.kind != "resource_rate":
            raise ValueError(
                "joint_service_does_not_identify_internal_resource_counterfactual")
        new_effective = _number(effective_rate, "counterfactual_effective_rate",
                                positive=True)
        return self._rate_intervention(
            work, new_effective, basis="explicit_effective_rate_intervention",
            assumption=("the steady-state effective rate is set directly; "
                        "fixed_us is unchanged and no nominal multiplier is "
                        "applied"),
            multiplier_source=None)

    def to_dict(self):
        return {
            "schema": RESOURCE_SERVICE_SCHEMA,
            "id": self.service_id,
            "kind": self.kind,
            "scope": self.scope,
            "resource": self.resource,
            "resources": list(self.resources),
            "phase": self.phase,
            "work_unit": self.work_unit,
            "boundary": self.boundary,
            "startup_us": self.startup_us,
            "response_us": self.response_us,
            "fixed_us": self.fixed_us,
            "effective_rate": self.effective_rate,
            "rate_scope": self.rate_scope,
            "active_units": self.active_units,
            "aggregate_rate": self.aggregate_rate,
            "per_owner_rate": self.per_owner_rate,
            "nominal_rate": self.nominal_rate,
            "nominal_multiplier": self.nominal_multiplier,
            "conditions": deepcopy(self.conditions),
            "optional_conditions": deepcopy(self.optional_conditions),
            "identification": deepcopy(self.identification),
            "provenance": deepcopy(self.provenance),
            "rate_representation": deepcopy(self.rate_representation),
            "normalization": deepcopy(self.normalization),
            "formula": self.formula,
            "model_id": self.model_id,
        }


def _normalized_service_from_record(record, *, hardware=None):
    """Reload a normalized export without applying its multiplier again."""
    if hardware:
        conditions = record.get("conditions") or {}
        declared_hardware = conditions.get("hardware") if isinstance(
            conditions, Mapping) else None
        if declared_hardware and str(declared_hardware) != str(hardware):
            raise ValueError("resource_service_hardware_mismatch")
    expected = {
        "id", "kind", "scope", "resources", "phase", "work_unit", "boundary",
        "startup_us", "response_us", "fixed_us", "effective_rate", "rate_scope",
        "active_units", "aggregate_rate", "per_owner_rate",
    }
    missing = sorted(expected - set(record))
    if missing:
        raise ValueError("resource_service_normalized_record_missing:" + ",".join(missing))
    normalization = dict(record.get("normalization") or {})
    normalization["reused"] = True
    normalization.setdefault("mode", "normalized_record")
    raw_resources = record["resources"]
    if isinstance(raw_resources, str):
        raw_resources = [raw_resources]
    return NormalizedResourceService(
        service_id=str(record["id"]),
        kind=str(record["kind"]),
        scope=str(record["scope"]),
        resources=tuple(str(item) for item in raw_resources),
        phase=str(record["phase"]),
        work_unit=str(record["work_unit"]),
        boundary=str(record["boundary"]),
        startup_us=_number(record["startup_us"], "startup_us"),
        response_us=_number(record["response_us"], "response_us"),
        fixed_us=_number(record["fixed_us"], "fixed_us"),
        effective_rate=_number(record["effective_rate"], "effective_rate", positive=True),
        rate_scope=str(record["rate_scope"]),
        active_units=_integer(record["active_units"], "active_units", minimum=1),
        aggregate_rate=_number(record["aggregate_rate"], "aggregate_rate", positive=True),
        per_owner_rate=_number(record["per_owner_rate"], "per_owner_rate", positive=True),
        nominal_rate=(None if record.get("nominal_rate") is None else
                      _number(record["nominal_rate"], "nominal_rate", positive=True)),
        nominal_multiplier=(None if record.get("nominal_multiplier") is None else
                            _number(record["nominal_multiplier"],
                                    "nominal_multiplier", positive=True)),
        conditions=record.get("conditions") or {},
        optional_conditions=record.get("optional_conditions") or {},
        identification=record.get("identification") or {},
        provenance=record.get("provenance") or {},
        rate_representation=record.get("rate_representation") or {},
        normalization=normalization,
    )


def normalize_resource_service(record, *, nominal_rates=None, hardware=None):
    """Normalize one declared service or reload a normalized service export.

    The declared supply must use exactly one of ``absolute_rate``,
    ``nominal_multiplier`` or ``nominal``.  ``effective_rate`` is computed once
    here; consumers receive only the normalized value.
    """
    if not isinstance(record, Mapping):
        raise ValueError("resource_service_requires_mapping")
    if record.get("schema") == RESOURCE_SERVICE_SCHEMA:
        normalization = record.get("normalization") or {}
        if isinstance(normalization, Mapping) and normalization.get("applied_once"):
            return _normalized_service_from_record(record, hardware=hardware)

    service_id = _text(record.get("id") or record.get("service_id"),
                       "service_id", default="resource-service")
    scope = _text(record.get("scope"), "scope")
    if scope not in _SERVICE_SCOPES:
        raise ValueError("resource_service_scope_must_be_independent_or_joint")
    resources = _resource_list(record)
    if scope == "independent_resource" and len(resources) != 1:
        raise ValueError("independent_resource_service_requires_exactly_one_resource")
    phase = _text(record.get("phase"), "phase")
    work_unit = _text(record.get("work_unit"), "work_unit")
    boundary = _text(record.get("boundary"), "boundary")
    if boundary not in _SERVICE_BOUNDARIES:
        raise ValueError(
            "resource_service_boundary_must_be_resource_phase_or_device_kernel")
    startup_us = _number(record.get("startup_us", 0), "startup_us")
    response_us = _number(record.get("response_us", 0), "response_us")
    fixed_us = startup_us + response_us
    rate_scope = _text(record.get("rate_scope", "aggregate"), "rate_scope")
    if rate_scope not in _RATE_SCOPES:
        raise ValueError("resource_service_rate_scope_must_be_aggregate_or_per_owner")
    active_units = _integer(record.get("active_units", 1), "active_units", minimum=1)

    supply = record.get("supply")
    if not isinstance(supply, Mapping):
        raise ValueError("resource_service_requires_a_supply_mapping")
    unknown_supply = set(supply) - {"kind", "rate", "nominal_rate",
                                   "multiplier", "nominal_unit", "source"}
    if unknown_supply:
        raise ValueError("resource_service_supply_unknown_fields:"
                         + ",".join(sorted(str(item) for item in unknown_supply)))
    supply_kind = _text(supply.get("kind"), "supply_kind")
    if supply_kind not in _SUPPLY_KINDS:
        raise ValueError("resource_service_supply_kind_unknown")
    nominal_unit = supply.get("nominal_unit", work_unit)
    if nominal_unit != work_unit:
        raise ValueError("resource_service_nominal_rate_unit_mismatch")
    declared_nominal = supply.get("nominal_rate")
    if declared_nominal is not None:
        declared_nominal = _number(declared_nominal, "nominal_rate", positive=True)
    referenced_nominal = None
    if nominal_rates is not None and len(resources) == 1:
        referenced_nominal = resolve_nominal_rate(
            resources[0], nominal_rates, work_unit=work_unit, hardware=hardware,
            rate_scope=rate_scope)
    if (declared_nominal is not None and referenced_nominal is not None
            and not math.isclose(declared_nominal, referenced_nominal,
                                 rel_tol=1e-9, abs_tol=0.0)):
        raise ValueError("resource_service_nominal_rate_conflict")
    nominal_rate = declared_nominal if declared_nominal is not None else referenced_nominal

    if supply_kind == "absolute_rate":
        if supply.get("multiplier") is not None:
            raise ValueError("resource_service_supply_representation_ambiguous")
        declared_rate = _number(supply.get("rate"), "rate", positive=True)
        effective_rate = declared_rate
        nominal_multiplier = (declared_rate / nominal_rate
                              if nominal_rate is not None else None)
    elif supply_kind == "nominal_multiplier":
        if supply.get("rate") is not None:
            raise ValueError("resource_service_supply_representation_ambiguous")
        if nominal_rate is None:
            raise ValueError("resource_service_multiplier_requires_nominal_rate")
        nominal_multiplier = _number(supply.get("multiplier"), "multiplier", positive=True)
        effective_rate = nominal_rate * nominal_multiplier
    else:
        if supply.get("rate") is not None or supply.get("multiplier") is not None:
            raise ValueError("resource_service_supply_representation_ambiguous")
        if nominal_rate is None:
            raise ValueError("resource_service_nominal_supply_requires_nominal_rate")
        nominal_multiplier = 1.0
        effective_rate = nominal_rate
    if not math.isfinite(effective_rate) or effective_rate <= 0:
        raise ValueError("resource_service_effective_rate_requires_positive_finite")
    if nominal_rate is None:
        multiplier_source = None
    elif supply_kind == "nominal_multiplier":
        multiplier_source = "declared"
    elif supply_kind == "nominal":
        multiplier_source = "identity"
    else:
        multiplier_source = "inferred_from_effective_rate_and_nominal"

    conditions = _plain_mapping(record.get("conditions"), "conditions")
    optional_conditions = _plain_mapping(
        record.get("optional_conditions"), "optional_conditions")
    if hardware:
        declared_hardware = conditions.get("hardware")
        if declared_hardware and str(declared_hardware) != str(hardware):
            raise ValueError("resource_service_hardware_mismatch")
        conditions["hardware"] = str(hardware)

    identification = _plain_mapping(record.get("identification"), "identification")
    resource_identified = identification.get("resource_identified", False)
    if not isinstance(resource_identified, bool):
        raise ValueError("resource_service_resource_identified_requires_boolean")
    if scope == "joint_phase" and resource_identified:
        raise ValueError("joint_service_cannot_identify_an_internal_resource")
    status = identification.get("status")
    if status is None:
        if scope == "joint_phase":
            status = "joint_identified"
        elif supply_kind == "nominal":
            status = "nominal_specification"
        elif resource_identified:
            status = "identified"
        else:
            status = "declared_effective_service"
    identification["status"] = _text(status, "identification_status")
    if identification["status"] == "identified" and not resource_identified:
        raise ValueError("resource_service_identified_status_requires_identified_resource")
    if identification["status"] == "joint_identified" and scope != "joint_phase":
        raise ValueError("resource_service_joint_status_requires_joint_scope")
    if (identification["status"] == "nominal_specification"
            and supply_kind != "nominal"):
        raise ValueError("resource_service_nominal_status_requires_nominal_supply")
    identification["resource_identified"] = resource_identified
    identification["startup_response_separable"] = bool(
        identification.get("startup_response_separable", False))
    identification["fixed_components"] = str(
        identification.get("fixed_components")
        or ("startup_plus_response" if response_us else "startup_only"))
    for key in ("train_work", "heldout_work"):
        values = identification.get(key)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"resource_service_{key}_requires_list")
        identification[key] = sorted(
            _number(item, key, positive=True) for item in values)

    provenance = _plain_mapping(record.get("provenance"), "provenance")
    if scope == "joint_phase":
        parameter_class = "joint_phase_service"
        rate_interpretation = (
            "joint_phase_service_rate_not_an_isolated_internal_resource_rate")
    elif supply_kind == "nominal":
        parameter_class = "nominal_specification"
        rate_interpretation = "declared_nominal_rate_not_a_calibrated_service"
    else:
        parameter_class = "effective_resource_service"
        rate_interpretation = (
            "identified_effective_resource_service"
            if resource_identified else
            "declared_effective_resource_service_not_identified_as_isolated_peak")
    declared_class = provenance.get("parameter_class")
    if declared_class is not None and declared_class != parameter_class:
        raise ValueError("resource_service_parameter_class_conflict")
    provenance["parameter_class"] = parameter_class
    provenance.setdefault("rate_interpretation", rate_interpretation)
    provenance["normalized_rate_interpretation"] = rate_interpretation

    if rate_scope == "aggregate":
        aggregate_rate = effective_rate
        per_owner_rate = effective_rate / active_units
    else:
        aggregate_rate = effective_rate * active_units
        per_owner_rate = effective_rate
    normalization = {
        "applied_once": True,
        "mode": "declared_to_effective",
        "input_kind": supply_kind,
        "declared_rate": supply.get("rate"),
        "nominal_rate": nominal_rate,
        "multiplier": nominal_multiplier,
        "multiplier_source": multiplier_source,
        "effective_rate": effective_rate,
        "rate_scope": rate_scope,
        "fixed_us": fixed_us,
        "fixed_components": ["startup_us", "response_us"],
        "use": ("duration_us = fixed_us + work * 1e6 / effective_rate; "
                "do not add startup_us or response_us again"),
        "fixed_term_rule": ("fixed_us is charged once; do not add another "
                            "occupancy/tail term already included in it"),
        "note": ("effective rate resolved once; do not reapply the nominal "
                 "rate or multiplier in a consumer"),
    }
    return NormalizedResourceService(
        service_id=service_id,
        kind=("joint_phase" if scope == "joint_phase" else "resource_rate"),
        scope=scope,
        resources=resources,
        phase=phase,
        work_unit=work_unit,
        boundary=boundary,
        startup_us=startup_us,
        response_us=response_us,
        fixed_us=fixed_us,
        effective_rate=effective_rate,
        rate_scope=rate_scope,
        active_units=active_units,
        aggregate_rate=aggregate_rate,
        per_owner_rate=per_owner_rate,
        nominal_rate=nominal_rate,
        nominal_multiplier=nominal_multiplier,
        conditions=conditions,
        optional_conditions=optional_conditions,
        identification=identification,
        provenance=provenance,
        rate_representation={
            "input_kind": supply_kind,
            "declared_rate": supply.get("rate"),
            "nominal_rate": nominal_rate,
            "multiplier": nominal_multiplier,
            "multiplier_source": multiplier_source,
            "effective_rate": effective_rate,
            "work_unit": work_unit,
            "rate_scope": rate_scope,
        },
        normalization=normalization,
    )


__all__ = [
    "NormalizedResourceService",
    "RESOURCE_SERVICE_BINDING_SCHEMA",
    "RESOURCE_SERVICE_PROFILE_SCHEMA",
    "RESOURCE_SERVICE_SCHEMA",
    "normalize_resource_service",
    "resolve_nominal_rate",
    "select_resource_service",
    "select_resource_service_candidate",
    "service_conditions_match",
    "sparse_mte1_bytes",
]
