"""Consume normalized T06 resource/phase services as ordinary recipes.

[Role]
  Turn one accepted ``NormalizedResourceService`` formula into the existing
  ``ResourceService`` recipe boundary.  The formula is evaluated on the
  declared resource work for every shape; it is not an operator-shape lookup
  and it does not set a whole-operator target duration.

[Invariants]
  * Only one explicitly declared resource phase is constrained per occurrence.
  * The T06 effective rate and fixed term are resolved once by its normalizer.
  * Conditions declared by the service are checked against the actual
    hardware/runtime identity before the service is used.
  * The service changes resource supply; other work and the Engine remain the
    ordinary execution path.

[Timeouts & Locks]
  Pure construction and one formula evaluation per occurrence.

[Side-Effect Policy]
  Builds an existing ``ResourceService``; no I/O, device work or global state.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import json
from typing import Any, Mapping

from mindie_sim.observations.calibration import service_conditions_match
from mindie_sim.observations.resource_services import NormalizedResourceService
from mindie_sim.prediction.service_regime_domains.services import (
    ResourceService, ServiceCurve, ServiceEstimate,
)


def conditions_context(context, identity=None) -> dict[str, Any]:
    identity = dict(context.service_identity if identity is None else identity or {})
    hardware = dict(identity.get("hardware") or {})
    # Engine rates are work-units/ms keyed by physical resource.  T06
    # conditions use declared roles and the hardware units (for example
    # ``rates.hbm`` in B/s).  Publish both views, with the role view converted
    # exactly once from the same physical rate.
    rates = dict(context.rates or {})
    for role, physical in dict(context.resource_names or {}).items():
        if physical in rates:
            rates[role] = float(rates[physical]) * 1000.0
    return {
        **identity,
        "hardware": hardware.get("key") or hardware.get("architecture") or "",
        "architecture": hardware.get("architecture") or "",
        "rates": rates,
        "resource_units": dict(context.resource_units or {}),
        "work_units": dict(context.work_units or {}),
        "compute_peaks": dict(context.compute_peaks or {}),
        "pool_capacities": dict(context.pool_capacities or {}),
    }


class FormulaServiceCurve(ServiceCurve):
    """A T06 affine service evaluated for every declared work size.

    ``ServiceCurve`` normally interpolates measured knots and rejects work
    outside them.  This thin subclass keeps the accepted normalized formula
    (``fixed_us + work * 1e6 / effective_rate``) and hands it to the existing
    ``ResourceService`` cap/placement logic.  It is not a target-latency
    envelope: the Engine still services the declared resource work and keeps
    the faster physical supply when it is the binding term.
    """

    def __init__(self, service: NormalizedResourceService, identity: Mapping):
        if service.scope != "independent_resource":
            raise ValueError(
                "user_profile_joint_phase_requires_an_explicit_phase_boundary")
        fixed_ms = service.fixed_us / 1000.0
        knots = ((1.0, service.duration_us(1.0) / 1000.0),
                 (2.0, service.duration_us(2.0) / 1000.0))
        super().__init__(
            service.resource, service.work_unit, identity, service.boundary,
            service.active_units, fixed_ms, knots,
            provenance=service.provenance,
        )
        object.__setattr__(self, "service", service)

    def estimate(self, work: float) -> ServiceEstimate | None:
        if work <= 0:
            return None
        # ``work`` is the aggregate declared resource demand carried by the
        # Engine.  T06's per-owner representation expects per-owner work, so
        # convert once before evaluating its single normalized formula.
        scoped_work = (work / self.service.active_units
                       if self.service.rate_scope == "per_owner" else work)
        duration_us = self.service.duration_us(
            scoped_work, work_scope=self.service.rate_scope)
        service_us = duration_us - self.service.fixed_us
        if service_us <= 0:
            return None
        return ServiceEstimate(
            work, self.startup_ms, service_us / 1000.0,
            "t06_normalized_resource_service_formula",
        )


class ProfileServiceRecipe(ResourceService):
    """Bind one T06 service at lowering time on the ordinary Engine.

    It subclasses ``ResourceService`` so the existing invocation policy sees the
    inclusive resource boundary and rejects a second opaque startup.  The
    lowering-time subclass only replaces the identity with the actual placement;
    the T06 normalizer remains the sole price authority.
    """

    def __init__(self, service: NormalizedResourceService):
        if not isinstance(service, NormalizedResourceService):
            raise TypeError("user_profile_service_requires_normalized_resource_service")
        self.service = service
        super().__init__(FormulaServiceCurve(service, {}))

    def lower(self, context):
        conditions = conditions_context(context)
        if not service_conditions_match(self.service, conditions)["applies"]:
            raise ValueError(
                "user_profile_service_conditions_do_not_match:"
                + self.service.service_id)
        # T04 now attaches the selected transport option object to the
        # occurrence identity.  It is provenance, not a T06 condition, and the
        # ServiceCurve identity is persisted as canonical JSON.  Normalize it
        # to plain data for the service boundary while preserving every field.
        identity = dict(context.service_identity)
        for key, value in tuple(identity.items()):
            if is_dataclass(value) and not isinstance(value, type):
                identity[key] = asdict(value)
        identity = json.loads(json.dumps(
            identity, sort_keys=True, allow_nan=False))
        service_context = replace(context, service_identity=identity)
        curve = FormulaServiceCurve(self.service, identity)
        return ResourceService(curve).lower(service_context)


__all__ = ["FormulaServiceCurve", "ProfileServiceRecipe", "conditions_context"]
