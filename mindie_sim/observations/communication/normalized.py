"""Normalized resource-service stage binding and ServiceCurve adaptation.

[Role] Adapt T06 normalized resource services to engine ``ServiceCurve`` and
  ``ResourceService`` objects and bind selected public stages with explicit
  work-scope and condition checking.
[Invariants] A joint phase keeps its resource set and never compresses to a
  single link cap. A child/phase resource without a physical byte mapping is
  rejected rather than renamed to ``link``. Fixed startup is charged once.
[Timeouts & Locks] Pure in-memory computation; no locks, timers, or I/O.
[Side-Effect Policy] Pure functions; never mutates input service records or
  global state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Mapping

from mindie_sim.prediction.services import ResourceService, ServiceCurve
from mindie_sim.schema.common import ModelingError

DEFAULT_NORMALIZED_WORK_POINTS = (1.0, 1e15)
_NORMALIZED_PHYSICAL_RESOURCES = frozenset({
    "link", "hbm", "mte1", "mte2", "mte3", "fixpipe",
})

__all__ = [
    "DEFAULT_NORMALIZED_WORK_POINTS",
    "NormalizedServiceStage",
    "build_normalized_service_recipe",
    "build_normalized_service_stage",
    "normalized_service_duration_ms",
    "service_curve_from_normalized",
]


@dataclass(frozen=True)
class NormalizedServiceStage:
    """One selected public stage bound to a T06 normalized service.

    ``kind`` is ``independent_resource`` or ``joint_phase``.  A joint phase
    keeps its resource set and never becomes one link cap.  A child/phase
    resource that has no physical byte mapping is rejected before this record.
    """

    kind: str
    service_id: str
    resources: tuple[str, ...]
    physical_resource: str | None
    phase: str
    work_unit: str
    boundary: str
    work: float
    work_scope: str
    active_units: int
    fixed_us: float
    duration_us: float
    owner_private_rate: float
    aggregate_rate: float
    stage_cap_work_per_ms: float
    aggregate_cap_work_per_ms: float
    work_total_aggregate: float
    conditions_bound: bool
    production_eligible: bool
    resource_service: ResourceService | None = None
    conditions: Mapping = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    def to_record(self):
        return {
            "kind": self.kind, "service_id": self.service_id,
            "resources": list(self.resources),
            "physical_resource": self.physical_resource,
            "phase": self.phase, "work_unit": self.work_unit,
            "boundary": self.boundary, "work": self.work,
            "work_scope": self.work_scope,
            "active_units": self.active_units,
            "fixed_us": self.fixed_us, "duration_us": self.duration_us,
            "owner_private_rate": self.owner_private_rate,
            "aggregate_rate": self.aggregate_rate,
            "stage_cap_work_per_ms": self.stage_cap_work_per_ms,
            "aggregate_cap_work_per_ms": self.aggregate_cap_work_per_ms,
            "work_total_aggregate": self.work_total_aggregate,
            "conditions_bound": self.conditions_bound,
            "production_eligible": self.production_eligible,
            "conditions": deepcopy(self.conditions),
            "provenance": deepcopy(self.provenance),
        }


def _normalized_service(service):
    from mindie_sim.observations.resource_services import (
        normalize_resource_service,
    )
    if hasattr(service, "duration_us") and hasattr(service, "effective_rate"):
        return service
    return normalize_resource_service(service)


def _normalized_work_scope(normalized, work_scope):
    # The default production adapter consumes an aggregate stage work amount.
    # A per-owner representation is only used when the real split work/owners
    # are supplied explicitly.  Stored rate_scope decides normalization, not
    # the stage work coordinate.
    del normalized
    scope = work_scope or "aggregate"
    if scope not in {"aggregate", "per_owner"}:
        raise ModelingError("normalized_service_work_scope_unknown:" + str(scope))
    return scope


def _normalized_rate_for_scope(normalized, work_scope):
    rate = (normalized.aggregate_rate if work_scope == "aggregate"
            else normalized.per_owner_rate)
    if rate <= 0 or not math.isfinite(rate):
        raise ModelingError("normalized_service_rate_not_finite")
    return rate


def _normalized_conditions_bound(normalized, conditions_context):
    if conditions_context is None:
        return False
    from mindie_sim.observations.calibration import service_conditions_match
    matched = service_conditions_match(normalized, conditions_context)
    if not matched["applies"]:
        failed = [check["path"] for check in matched["checks"]
                  if not check["matched"]]
        raise ModelingError(
            "normalized_service_conditions_mismatch:" + ",".join(failed))
    return True


def normalized_service_duration_ms(service, work, *, work_scope=None):
    """Apply T06's owner-aware formula exactly once."""
    normalized = _normalized_service(service)
    scope = _normalized_work_scope(normalized, work_scope)
    return normalized.duration_us(work, work_scope=scope) / 1000.0


def _normalized_service_curve(service, *, identity, resource, work_scope,
                              work_points, provenance, conditions_bound,
                              phase=None):
    normalized = _normalized_service(service)
    scope = _normalized_work_scope(normalized, work_scope)
    rate = _normalized_rate_for_scope(normalized, scope)
    points = tuple(float(point) for point in work_points)
    if (len(points) < 2
            or any(point <= 0 or not math.isfinite(point) for point in points)
            or any(right <= left for left, right in zip(points, points[1:]))):
        raise ValueError("normalized_service_requires_ordered_positive_work_points")
    knots = tuple(
        (point, (normalized.fixed_us + point * 1e6 / rate) / 1000.0)
        for point in points)
    details = {
        "normalized_service_id": normalized.model_id,
        "normalized_service_kind": normalized.kind,
        "normalized_service_scope": normalized.scope,
        "normalized_phase": normalized.phase,
        "normalized_work_unit": normalized.work_unit,
        "normalized_rate_scope": normalized.rate_scope,
        "service_work_scope": scope,
        "normalized_effective_rate": normalized.effective_rate,
        "owner_private_rate": normalized.per_owner_rate,
        "aggregate_rate": normalized.aggregate_rate,
        "aggregate_cap_work_per_ms": normalized.aggregate_rate / 1000.0,
        "normalized_fixed_us": normalized.fixed_us,
        "fixed_source": (normalized.identification or {}).get("fixed_source"),
        "fixed_components": (normalized.identification or {}).get(
            "fixed_components"),
        "fixed_charged": "once_in_ResourceService_setup",
        "normalized_conditions": dict(normalized.conditions),
        "normalized_provenance": dict(normalized.provenance),
        "conditions_bound": bool(conditions_bound),
        "whole_call_floor": False,
        "production_stage": bool(conditions_bound),
        "diagnostic_only": not bool(conditions_bound),
        "formula": "T_us = fixed_us + work * 1e6 / scoped_rate",
    }
    if phase is not None and phase != normalized.phase:
        raise ModelingError("normalized_service_phase_mismatch")
    if provenance:
        details.update(provenance)
    return ServiceCurve(
        resource, normalized.work_unit, identity, normalized.boundary,
        normalized.active_units, normalized.fixed_us / 1000.0, knots, details)


def service_curve_from_normalized(service, *, identity, resource=None,
                                  boundary=None, active_units=None,
                                  work_points=DEFAULT_NORMALIZED_WORK_POINTS,
                                  provenance=None, conditions_context=None,
                                  work_scope=None):
    """Adapt one T06 independent service to ServiceCurve.

    A joint phase or a child/phase resource without a real physical byte
    mapping is rejected; renaming it to ``link`` is not accepted.  Aggregate
    vs per-owner work is selected explicitly and evaluated by T06.
    """
    normalized = _normalized_service(service)
    if normalized.scope != "independent_resource":
        raise ModelingError("normalized_service_joint_phase_requires_joint_stage")
    if len(normalized.resources) != 1:
        raise ModelingError("normalized_service_independent_requires_one_resource")
    expected = normalized.resources[0]
    resource = resource or expected
    if resource != expected:
        raise ModelingError("normalized_service_resource_mismatch")
    if resource not in _NORMALIZED_PHYSICAL_RESOURCES:
        raise ModelingError("normalized_service_resource_not_physical")
    if boundary is not None and boundary != normalized.boundary:
        raise ModelingError("normalized_service_boundary_mismatch")
    units = normalized.active_units if active_units is None else int(active_units)
    if units != normalized.active_units:
        raise ModelingError("normalized_service_active_units_mismatch")
    scope = _normalized_work_scope(normalized, work_scope)
    conditions_bound = _normalized_conditions_bound(normalized, conditions_context)
    return _normalized_service_curve(
        normalized, identity=identity, resource=resource, work_scope=scope,
        work_points=work_points, provenance=provenance,
        conditions_bound=conditions_bound, phase=normalized.phase)


def build_normalized_service_stage(service, *, identity, stage,
                                   conditions_context=None,
                                   work_points=DEFAULT_NORMALIZED_WORK_POINTS,
                                   provenance=None, require_conditions=False):
    """Bind one selected public Stage's work/owner/role to a T06 service."""
    normalized = _normalized_service(service)
    stage = dict(stage or {})
    for key in ("work", "work_scope", "active_units", "phase", "work_unit",
                "boundary"):
        if key not in stage:
            raise ValueError("normalized_service_stage_requires_" + key)
    work = float(stage["work"])
    if not math.isfinite(work) or work <= 0:
        raise ValueError("normalized_service_stage_work_positive")
    scope = _normalized_work_scope(normalized, stage["work_scope"])
    active = int(stage["active_units"])
    if active != normalized.active_units:
        raise ModelingError("normalized_service_active_units_mismatch")
    if str(stage["phase"]) != normalized.phase:
        raise ModelingError("normalized_service_phase_mismatch")
    if str(stage["work_unit"]) != normalized.work_unit:
        raise ModelingError("normalized_service_work_unit_mismatch")
    if str(stage["boundary"]) != normalized.boundary:
        raise ModelingError("normalized_service_boundary_mismatch")
    conditions_bound = _normalized_conditions_bound(normalized, conditions_context)
    if require_conditions and not conditions_bound:
        raise ModelingError("normalized_service_conditions_required_for_production")
    duration_us = normalized.duration_us(work, work_scope=scope)
    fixed_us = normalized.fixed_us
    service_ms = duration_us - fixed_us
    if service_ms <= 0 or not math.isfinite(service_ms):
        raise ModelingError("normalized_service_stage_requires_positive_service")
    stage_cap = work * 1000.0 / service_ms
    aggregate_cap = normalized.aggregate_rate / 1000.0
    work_total = work if scope == "aggregate" else work * active
    common = dict(
        service_id=normalized.service_id, phase=normalized.phase,
        work_unit=normalized.work_unit, boundary=normalized.boundary,
        work=work, work_scope=scope, active_units=active,
        fixed_us=fixed_us, duration_us=duration_us,
        owner_private_rate=normalized.per_owner_rate,
        aggregate_rate=normalized.aggregate_rate,
        stage_cap_work_per_ms=stage_cap,
        aggregate_cap_work_per_ms=aggregate_cap,
        work_total_aggregate=work_total,
        conditions_bound=conditions_bound,
        conditions=dict(normalized.conditions),
        provenance={
            "normalized_service_id": normalized.model_id,
            "normalized_service_kind": normalized.kind,
            "normalized_service_scope": normalized.scope,
            "normalized_fixed_us": fixed_us,
            "fixed_source": (normalized.identification or {}).get("fixed_source"),
            "fixed_components": (normalized.identification or {}).get(
                "fixed_components"),
            "fixed_charged": "once_in_ResourceService_setup",
            "scoped_rate": _normalized_rate_for_scope(normalized, scope),
            "whole_call_floor": False,
            "selected_stage": stage,
            **(dict(provenance) if provenance else {}),
        },
    )
    if normalized.scope == "joint_phase":
        resources = tuple(str(item) for item in (stage.get("resources") or ()))
        if set(resources) != set(normalized.resources):
            raise ModelingError("normalized_service_joint_phase_resource_mismatch")
        if any(item not in _NORMALIZED_PHYSICAL_RESOURCES for item in resources):
            raise ModelingError(
                "normalized_service_joint_phase_resource_not_physical")
        return NormalizedServiceStage(
            kind="joint_phase", resources=resources, physical_resource=None,
            resource_service=None,
            production_eligible=bool(conditions_bound), **common)
    if normalized.scope != "independent_resource":
        raise ModelingError("normalized_service_scope_unknown")
    resource = stage.get("resource")
    if resource != normalized.resources[0]:
        raise ModelingError("normalized_service_resource_mismatch")
    if resource not in _NORMALIZED_PHYSICAL_RESOURCES:
        raise ModelingError("normalized_service_resource_not_physical")
    if "resources" in stage and tuple(stage["resources"]) != normalized.resources:
        raise ModelingError("normalized_service_resource_mismatch")
    curve = _normalized_service_curve(
        normalized, identity=identity, resource=resource, work_scope=scope,
        work_points=work_points, provenance=common["provenance"],
        conditions_bound=conditions_bound, phase=normalized.phase)
    return NormalizedServiceStage(
        kind="independent_resource", resources=normalized.resources,
        physical_resource=resource, resource_service=ResourceService(curve),
        production_eligible=bool(conditions_bound), **common)


def build_normalized_service_recipe(service, *, identity, resource=None,
                                    boundary=None, active_units=None,
                                    work_points=DEFAULT_NORMALIZED_WORK_POINTS,
                                    provenance=None, conditions_context=None,
                                    work_scope=None, require_conditions=True,
                                    stage=None):
    """Return ResourceService for one independent T06 service stage.

    When ``stage`` is supplied it is the same selected Stage as the public
    execution binding; the returned recipe is its ``resource_service``.
    """
    if stage is not None:
        bound = build_normalized_service_stage(
            service, identity=identity, stage=stage,
            conditions_context=conditions_context, work_points=work_points,
            provenance=provenance, require_conditions=require_conditions)
        if bound.resource_service is None:
            raise ModelingError("normalized_service_joint_phase_requires_joint_stage")
        return bound.resource_service
    if require_conditions and conditions_context is None:
        raise ModelingError("normalized_service_conditions_required_for_production")
    curve = service_curve_from_normalized(
        service, identity=identity, resource=resource, boundary=boundary,
        active_units=active_units, work_points=work_points,
        provenance=provenance, conditions_context=conditions_context,
        work_scope=work_scope)
    return ResourceService(curve)
