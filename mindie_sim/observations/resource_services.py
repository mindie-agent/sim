"""Normalized resource/phase services and scoped C7 service selection.

[Role] Resolve a declared nominal, effective or joint resource service exactly
  once, and turn one profiler `resource_services` record into a query-geometry
  service time for the ordinary sparse attention consumer.
[Invariants] A service has one resource/phase boundary and one normalized
  effective rate.  Absolute rate, nominal specification and nominal multiplier
  are mutually exclusive supply representations; the effective rate is derived
  once and a normalized record is never multiplied again.  Startup and response
  are separate declared terms but enter the formula once.  A joint-phase window
  predicts its own work and never claims an internal per-resource rate or a
  per-resource counterfactual.  The C7 record carries the R64 source identity,
  native kernel, dtype, hardware, owner count, measured critical bytes and
  measured per-critical-owner rate; the query must be the same
  source/native/hardware/format, a saturated prefill class and owner count.
[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Pure result; no case or metadata mutation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .resource_services_rates import _RATE_SCOPES, _number, _text, resolve_nominal_rate
from .resource_services_selection import (
    c7_applicability,
    c7_applicability_matches,
    identity_digest,
    select_resource_service,
    sparse_mte1_bytes,
)


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
    "sparse_mte1_bytes",
]
