"""Communication capture schemas, golden constants and normalization.

[Role] Own normalized collective row dataclasses, A5 golden constants, and normalization.
[Invariants] Zero upward imports inside communication package.
[Timeouts & Locks] Pure in-memory definitions; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Mapping

from mindie_sim.prediction.service_regime_domains.services import ResourceService, ServiceCurve
from mindie_sim.schema.common import ModelingError

DEFAULT_NORMALIZED_WORK_POINTS = (1.0, 1e15)
_NORMALIZED_PHYSICAL_RESOURCES = frozenset({
    "link", "hbm", "mte1", "mte2", "mte3", "fixpipe",
})



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


from bisect import bisect_left

A5_HARDWARE_KEY = "Ascend950DT_9582:ub"
A5_SKU = "Ascend950DT_9582"
A5_ARCH_FAMILY = "A5"


EMBEDDED_A5_CAMPAIGNS = {
    "comm-rack06": {
        "label": "comm-rack06", "kind": "a5_collective_calibration", "status": "completed",
        "canary": "all_ranks_matched", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "stack": "rack06", "cann": "9.1.0",
        "date": "2026-10-21", "dtype": "bf16", "observed_expansion_mode": "not_observed",
        "placement": "a5-ub-mesh/rack06/intra-node", "nodes": ("rack06",),
        "requested_expansion_mode": "unspecified", "source": "capture://mindie-sim-data/a5/2026-10-21/comm-rack06/a5-comm-rack06-calibration.json",
        "series": (
            ("all_gather", 2, "intra_node_ub", None, ((1024, 0.050598), (4096, 0.049899), (16384, 0.049841), (65536, 0.051102), (262144, 0.052241), (1048576, 0.054034), (4194304, 0.113933), (16777216, 0.377593), (67108864, 1.462068), (134217728, 2.910362))),
            ("all_gather", 4, "intra_node_ub", None, ((1024, 0.056646), (4096, 0.057046), (16384, 0.057128), (65536, 0.057845), (262144, 0.059926), (1048576, 0.062663), (4194304, 0.119794), (16777216, 0.382312), (67108864, 1.468641), (134217728, 2.950437))),
            ("all_gather", 8, "intra_node_ub", None, ((1024, 0.080740), (4096, 0.082026), (16384, 0.082252), (65536, 0.083512), (262144, 0.085663), (1048576, 0.092175), (4194304, 0.131170), (16777216, 0.394093), (67108864, 1.517734), (134217728, 3.059270))),
            ("all_reduce", 2, "intra_node_ub", None, ((1024, 0.050758), (4096, 0.047830), (16384, 0.047731), (65536, 0.049085), (262144, 0.049331), (1048576, 0.051784), (4194304, 0.110267), (16777216, 0.376063), (67108864, 1.447595), (134217728, 2.868704))),
            ("all_reduce", 4, "intra_node_ub", None, ((1024, 0.055400), (4096, 0.054044), (16384, 0.053783), (65536, 0.054746), (262144, 0.056336), (1048576, 0.061233), (4194304, 0.122533), (16777216, 0.218719), (67108864, 0.780762), (134217728, 1.539571))),
            ("all_reduce", 8, "intra_node_ub", None, ((1024, 0.091568), (4096, 0.093453), (16384, 0.094894), (65536, 0.095555), (262144, 0.098381), (1048576, 0.107229), (4194304, 0.152148), (16777216, 0.189877), (67108864, 0.485184), (134217728, 0.946518))),
            ("all_to_all", 2, "intra_node_ub", None, ((1024, 0.050855), (4096, 0.050330), (16384, 0.052583), (65536, 0.052500), (262144, 0.055589), (1048576, 0.055884), (4194304, 0.069017), (16777216, 0.198776), (67108864, 0.737592), (134217728, 1.465435))),
            ("all_to_all", 4, "intra_node_ub", None, ((1024, 0.056785), (4096, 0.056545), (16384, 0.057168), (65536, 0.059709), (262144, 0.061374), (1048576, 0.064599), (4194304, 0.073074), (16777216, 0.120465), (67108864, 0.413982), (134217728, 0.817249))),
            ("all_to_all", 8, "intra_node_ub", None, ((1024, 0.102549), (4096, 0.102169), (16384, 0.102268), (65536, 0.103593), (262144, 0.105662), (1048576, 0.110865), (4194304, 0.121045), (16777216, 0.132739), (67108864, 0.273519), (134217728, 0.523422))),
            ("p2p_send_recv", 2, "intra_node_ub", "0->1", ((1024, 0.057611), (4096, 0.057137), (16384, 0.056567), (65536, 0.057854), (262144, 0.060202), (1048576, 0.059366), (4194304, 0.101420), (16777216, 0.347544), (67108864, 1.346780), (134217728, 2.695845))),
            ("p2p_send_recv", 4, "intra_node_ub", "0->1", ((1024, 0.055161), (4096, 0.055124), (16384, 0.054283), (65536, 0.054304), (262144, 0.054457), (1048576, 0.055498), (4194304, 0.101076), (16777216, 0.348060), (67108864, 1.349609), (134217728, 2.694642))),
            ("p2p_send_recv", 8, "intra_node_ub", "0->1", ((1024, 0.074796), (4096, 0.075102), (16384, 0.075990), (65536, 0.076425), (262144, 0.075687), (1048576, 0.078275), (4194304, 0.104849), (16777216, 0.351558), (67108864, 1.354754), (134217728, 2.716599))),
            ("reduce_scatter", 2, "intra_node_ub", None, ((1024, 0.047764), (4096, 0.048391), (16384, 0.048267), (65536, 0.048032), (262144, 0.049795), (1048576, 0.051545), (4194304, 0.111980), (16777216, 0.375897), (67108864, 1.468113), (134217728, 2.914158))),
            ("reduce_scatter", 4, "intra_node_ub", None, ((1024, 0.053205), (4096, 0.054090), (16384, 0.054304), (65536, 0.054827), (262144, 0.057186), (1048576, 0.062440), (4194304, 0.125447), (16777216, 0.404735), (67108864, 1.501118), (134217728, 2.974738))),
            ("reduce_scatter", 8, "intra_node_ub", None, ((1024, 0.094667), (4096, 0.095990), (16384, 0.095888), (65536, 0.096511), (262144, 0.099852), (1048576, 0.109664), (4194304, 0.156793), (16777216, 0.493925), (67108864, 1.794119), (134217728, 3.654562))),
        ),
        "single_anchors": (
            ("all_reduce", 8, "intra_node_ub", None, ((1024, 0.221124), (1048576, 0.250297))),
            ("all_gather", 8, "intra_node_ub", None, ((1024, 0.212235), (1048576, 0.234697))),
            ("reduce_scatter", 8, "intra_node_ub", None, ((1024, 0.231142), (1048576, 0.252197))),
            ("all_to_all", 8, "intra_node_ub", None, ((1024, 0.243767), (1048576, 0.246995))),
            ("p2p_send_recv", 8, "intra_node_ub", "0->1", ((1024, 0.146115), (1048576, 0.148707))),
            ("all_reduce", 4, "intra_node_ub", None, ((1024, 0.136011), (1048576, 0.159574))),
            ("all_gather", 4, "intra_node_ub", None, ((1024, 0.127443), (1048576, 0.147381))),
            ("reduce_scatter", 4, "intra_node_ub", None, ((1024, 0.136105), (1048576, 0.157033))),
            ("all_to_all", 4, "intra_node_ub", None, ((1024, 0.151496), (1048576, 0.153715))),
            ("p2p_send_recv", 4, "intra_node_ub", "0->1", ((1024, 0.099498), (1048576, 0.115558))),
            ("all_reduce", 2, "intra_node_ub", None, ((1024, 0.104110), (1048576, 0.123874))),
            ("all_gather", 2, "intra_node_ub", None, ((1024, 0.097608), (1048576, 0.119247))),
            ("reduce_scatter", 2, "intra_node_ub", None, ((1024, 0.104601), (1048576, 0.127464))),
            ("all_to_all", 2, "intra_node_ub", None, ((1024, 0.115710), (1048576, 0.127487))),
            ("p2p_send_recv", 2, "intra_node_ub", "0->1", ((1024, 0.101234), (1048576, 0.118528))),
        ),
    },
    "comm-rack08": {
        "label": "comm-rack08", "kind": "a5_collective_calibration", "status": "completed",
        "canary": "all_ranks_matched", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "stack": "rack08", "cann": "9.2.0",
        "date": "2026-10-21", "dtype": "bf16", "observed_expansion_mode": "not_observed",
        "placement": "a5-ub-mesh/rack08/intra-node", "nodes": ("rack08",),
        "requested_expansion_mode": "unspecified", "source": "capture://mindie-sim-data/a5/2026-10-21/comm-rack08/a5-comm-rack08-calibration.json",
        "series": (
            ("all_gather", 8, "intra_node_ub", None, ((1024, 0.057542), (4096, 0.051507), (16384, 0.060556), (65536, 0.050979), (262144, 0.053263), (1048576, 0.053907), (4194304, 0.119734), (16777216, 0.248473), (67108864, 0.950504), (134217728, 1.931796))),
            ("all_reduce", 2, "intra_node_ub", None, ((1024, 0.048851), (4096, 0.049171), (16384, 0.050588), (65536, 0.051537), (262144, 0.051087), (1048576, 0.054255), (4194304, 0.114074), (16777216, 0.377803), (67108864, 1.435099), (134217728, 2.873427))),
            ("all_reduce", 8, "intra_node_ub", None, ((1024, 0.062284), (4096, 0.062098), (16384, 0.063132), (65536, 0.064302), (262144, 0.066629), (1048576, 0.097578), (4194304, 0.108920), (16777216, 0.148519), (67108864, 0.545700), (134217728, 1.012155))),
            ("all_to_all", 2, "intra_node_ub", None, ((1024, 0.043024), (4096, 0.043510), (16384, 0.044084), (65536, 0.044185), (262144, 0.044748), (1048576, 0.047955), (4194304, 0.064382), (16777216, 0.195063), (67108864, 0.734224), (134217728, 1.461183))),
            ("all_to_all", 8, "intra_node_ub", None, ((1024, 0.058786), (4096, 0.050054), (16384, 0.051328), (65536, 0.051312), (262144, 0.052405), (1048576, 0.052844), (4194304, 0.059948), (16777216, 0.076900), (67108864, 0.245197), (134217728, 0.473576))),
            ("p2p_send_recv", 8, "intra_node_ub", "0->1", ((1024, 0.051051), (4096, 0.052369), (16384, 0.053105), (65536, 0.053310), (262144, 0.054773), (1048576, 0.056153), (4194304, 0.101179), (16777216, 0.346981), (67108864, 1.346166), (134217728, 2.695230))),
            ("reduce_scatter", 8, "intra_node_ub", None, ((1024, 0.057885), (4096, 0.056091), (16384, 0.056366), (65536, 0.057633), (262144, 0.057317), (1048576, 0.060866), (4194304, 0.147667), (16777216, 0.454569), (67108864, 1.653972), (134217728, 3.308766))),
        ),
        "single_anchors": (
            ("all_reduce", 8, "intra_node_ub", None, ((1024, 0.165914), (1048576, 0.212938))),
            ("all_gather", 8, "intra_node_ub", None, ((1024, 0.115449), (1048576, 0.137222))),
            ("reduce_scatter", 8, "intra_node_ub", None, ((1024, 0.128709), (1048576, 0.157305))),
            ("all_to_all", 8, "intra_node_ub", None, ((1024, 0.123795), (1048576, 0.124076))),
            ("p2p_send_recv", 8, "intra_node_ub", "0->1", ((1024, 0.101093), (1048576, 0.116892))),
            ("all_reduce", 2, "intra_node_ub", None, ((1024, 0.115710), (1048576, 0.130077))),
            ("all_to_all", 2, "intra_node_ub", None, ((1024, 0.094552), (1048576, 0.107434))),
        ),
    },
    "comm-w16": {
        "label": "comm-w16", "kind": "a5_collective_calibration", "status": "completed",
        "canary": "all_ranks_matched", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "stack": "rack06", "cann": "9.1.0",
        "date": "2026-10-22", "dtype": "bf16", "observed_expansion_mode": "not_observed",
        "placement": "a5-ub-mesh/rack06/inter-node/2-node-16rank", "nodes": ("node-0", "node-1"),
        "requested_expansion_mode": "unspecified", "source": "capture://mindie-sim-data/a5/2026-10-22/comm-w16/artifacts/a5-comm-w16-calibration.json",
        "series": (
            ("all_gather", 16, "cross_node_urma", None, ((1024, 0.075925), (4096, 0.077139), (16384, 0.080643), (65536, 0.089330), (262144, 0.110689), (1048576, 0.170982), (4194304, 0.370355), (16777216, 0.681688), (67108864, 2.695952), (134217728, 5.705244))),
            ("all_reduce", 16, "cross_node_urma", None, ((1024, 0.122519), (4096, 0.123304), (16384, 0.126710), (65536, 0.133490), (262144, 0.144056), (1048576, 0.164739), (4194304, 0.218155), (16777216, 0.333294), (67108864, 0.636665), (134217728, 0.980338))),
            ("all_to_all", 16, "cross_node_urma", None, ((1024, 0.203806), (4096, 0.202040), (16384, 0.204096), (65536, 0.205827), (262144, 0.209360), (1048576, 0.218068), (4194304, 0.234286), (16777216, 0.237027), (67108864, 0.300931), (134217728, 0.521780))),
            ("p2p_send_recv", 16, "cross_node_urma", "0->8", ((1024, 0.061526), (4096, 0.069573), (16384, 0.060116), (65536, 0.083580), (262144, 0.068095), (1048576, 0.068798), (4194304, 0.072223), (16777216, 0.091224), (67108864, 0.308109), (134217728, 0.619878))),
            ("p2p_send_recv", 16, "intra_node_ub", "0->1", ((1024, 0.066423), (4096, 0.066659), (16384, 0.066135), (65536, 0.066804), (262144, 0.062459), (1048576, 0.062873), (4194304, 0.102229), (16777216, 0.347678), (67108864, 1.351618), (134217728, 2.718014))),
            ("reduce_scatter", 16, "cross_node_urma", None, ((1024, 0.075152), (4096, 0.075359), (16384, 0.077166), (65536, 0.083789), (262144, 0.105831), (1048576, 0.175894), (4194304, 0.426934), (16777216, 0.808046), (67108864, 3.296926), (134217728, 7.048708))),
        ),
        "single_anchors": (
            ("all_reduce", 16, "cross_node_urma", None, ((1024, 0.376210), (1048576, 0.407557))),
            ("all_gather", 16, "cross_node_urma", None, ((1024, 0.315714), (1048576, 0.401103))),
            ("reduce_scatter", 16, "cross_node_urma", None, ((1024, 0.313759), (1048576, 0.384144))),
            ("all_to_all", 16, "cross_node_urma", None, ((1024, 0.399335), (1048576, 0.395260))),
            ("p2p_send_recv", 16, "intra_node_ub", "0->1", ((1024, 0.137419), (1048576, 0.131744))),
            ("p2p_send_recv", 16, "cross_node_urma", "0->8", ((1024, 0.121091), (1048576, 0.128381))),
        ),
    },
    "comm-w32": {
        "label": "comm-w32", "kind": "a5_collective_calibration", "status": "completed",
        "canary": "all_ranks_matched", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "stack": "rack06", "cann": "9.1.0",
        "date": "2026-10-22", "dtype": "bf16", "observed_expansion_mode": "not_observed",
        "placement": "a5-ub-mesh/rack06/inter-node/4-node-32rank", "nodes": ("node-0", "node-1", "node-2", "node-3"),
        "requested_expansion_mode": "unspecified", "source": "capture://mindie-sim-data/a5/2026-10-22/comm-w32/artifacts/a5-comm-w32-calibration.json",
        "series": (
            ("all_gather", 32, "cross_node_urma", None, ((1024, 0.114313), (4096, 0.111618), (16384, 0.113686), (65536, 0.124429), (262144, 0.165558), (1048576, 0.291884), (4194304, 0.518425), (16777216, 1.214689), (67108864, 4.846047), (134217728, 10.479134))),
            ("all_reduce", 32, "cross_node_urma", None, ((1024, 0.160401), (4096, 0.159696), (16384, 0.163973), (65536, 0.169966), (262144, 0.185857), (1048576, 0.212545), (4194304, 0.272354), (16777216, 0.394652), (67108864, 0.755183), (134217728, 1.036680))),
            ("all_to_all", 32, "cross_node_urma", None, ((1024, 0.339281), (4096, 0.352639), (16384, 0.354603), (65536, 0.357723), (262144, 0.361530), (1048576, 0.370592), (4194304, 0.377146), (16777216, 0.387694), (67108864, 0.476420), (134217728, 0.702319))),
            ("p2p_send_recv", 32, "cross_node_urma_far", "0->24", ((1024, 0.067301), (4096, 0.065435), (16384, 0.065767), (65536, 0.068113), (262144, 0.071937), (1048576, 0.076392), (4194304, 0.091687), (16777216, 0.099470), (67108864, 0.316095), (134217728, 0.641967))),
            ("p2p_send_recv", 32, "cross_node_urma", "0->8", ((1024, 0.067747), (4096, 0.073989), (16384, 0.077067), (65536, 0.076480), (262144, 0.076879), (1048576, 0.078880), (4194304, 0.070847), (16777216, 0.091770), (67108864, 0.307030), (134217728, 0.623592))),
            ("reduce_scatter", 32, "cross_node_urma", None, ((1024, 0.129680), (4096, 0.107657), (16384, 0.109174), (65536, 0.122996), (262144, 0.152992), (1048576, 0.290569), (4194304, 0.784332), (16777216, 1.418051), (67108864, 5.976353), (134217728, 12.902939))),
        ),
        "single_anchors": (
            ("all_reduce", 32, "cross_node_urma", None, ((1024, 0.428432), (1048576, 0.456942))),
            ("all_gather", 32, "cross_node_urma", None, ((1024, 0.340266), (1048576, 0.514703))),
            ("reduce_scatter", 32, "cross_node_urma", None, ((1024, 0.336648), (1048576, 0.492857))),
            ("all_to_all", 32, "cross_node_urma", None, ((1024, 0.553255), (1048576, 0.529210))),
            ("p2p_send_recv", 32, "cross_node_urma", "0->8", ((1024, 0.114919), (1048576, 0.119326))),
            ("p2p_send_recv", 32, "cross_node_urma_far", "0->24", ((1024, 0.132321), (1048576, 0.138508))),
        ),
    },
}

A5_NATIVE_GROUP2 = {
    "source": "capture://mindie-sim-data/calibration-20261107/a5/hccl-native/raw/parsed/native-summary.json",
    "manifest_sha256": "7ab3030edaf471d907fd83e1fec766ae37a4b2effde94411ee3bf14a975cc2b9",
    "platform": "A5 Ascend950DT intrahost",
    "stack": "rack08",
    "cann": "9.2.0",
    "mode": "aicpu",
    "dtype": "bf16",
    "split": "48 train / 45 heldout",
    "series": {
        "all_gather": (
            (32768, 0.069943), (786432, 0.074251), (1048576, 0.072051),
            (10485760, 0.111949), (33550336, 0.193240),
            (33558528, 0.187575), (50331648, 0.242885),
            (67108864, 0.294509),
        ),
        "reduce_scatter": (
            (65536, 0.058361), (1572864, 0.077590), (2097152, 0.081633),
            (20971520, 0.127892), (67100672, 0.203326),
            (67117056, 0.208005), (100663296, 0.250269),
            (134217728, 0.312730),
        ),
    },
    # Separate HcclLaunchAicpuKernel AI_CPU rows from the same conditions.
    # Qwen's TP kernels are AICPU, so this is the matching service boundary.
    "aicpu_series": {
        "all_gather": (
            (32768, 0.041113), (786432, 0.025880), (1048576, 0.040079),
            (10485760, 0.059318), (33550336, 0.060119),
            (33558528, 0.061353), (50331648, 0.060528),
            (67108864, 0.062541),
        ),
        "reduce_scatter": (
            (65536, 0.039239), (1572864, 0.044114), (2097152, 0.041995),
            (20971520, 0.062780), (67100672, 0.066477),
            (67117056, 0.063054), (100663296, 0.067298),
            (134217728, 0.064268),
        ),
    },
}


def a5_native_group2_curve_ms(collective: str, work_bytes: float,
                              service: str = "aicpu"):
    """Standalone group2 AllGather/ReduceScatter native service, rack08.

    These are the independent 2026-11-07 A5 intrahost native-wave medians for
    the exact group2 primitive, bf16, AICPU, rack08/CANN 9.2.0.  The axis is
    per-rank input bytes.  No extrapolation beyond the measured support and no
    MC2/AllToAll substitution is performed.
    """
    key = "aicpu_series" if str(service) == "aicpu" else "series"
    points = A5_NATIVE_GROUP2[key].get(str(collective))
    if not points or not work_bytes or work_bytes <= 0:
        return None
    if not points[0][0] <= work_bytes <= points[-1][0]:
        return None
    xs = [item[0] for item in points]
    right = bisect_left(xs, work_bytes)
    if points[right][0] == work_bytes:
        return points[right][1]
    left = right - 1
    x0, y0 = points[left]
    x1, y1 = points[right]
    return y0 + (y1 - y0) * (work_bytes - x0) / (x1 - x0)


EMBEDDED_A5_MC2_CAMPAIGNS = {
    "mc2-rack06-r1": {
        "label": "mc2-rack06-r1", "kind": "mc2_chain", "status": "completed", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "stack": "rack06", "date": "2026-10-21", "dtype": "bf16",
        "observed_expansion_mode": "not_observed", "requested_expansion_mode": "unspecified",
        "placement": "intra-node-ub", "nodes": ("node-0",),
        "source": "capture://mindie-sim-data/a5/2026-10-21/mc2/a5mc2-rack06-r1",
        "chains": (
            ("bf16", 2, 64, 512, 0.519084, 0.644950, 0.042783, 0.341566, 0.531807, 2.210635),
            ("bf16", 2, 64, 1024, 0.954289, 0.735582, 0.044903, 0.387475, 1.004513, 3.293601),
            ("bf16", 2, 128, 512, 0.523852, 1.143049, 0.047721, 0.612410, 0.536493, 3.073176),
            ("bf16", 2, 128, 1024, 0.949646, 1.284214, 0.047685, 0.661042, 0.980015, 4.176855),
            ("bf16", 4, 64, 1024, 0.346629, 0.374904, 0.049022, 0.205385, 0.352475, 1.401746),
            ("bf16", 4, 64, 2048, 0.568605, 0.870681, 0.047782, 0.460835, 0.592220, 2.754368),
            ("bf16", 4, 128, 1024, 0.339475, 0.646306, 0.047850, 0.341897, 0.325546, 1.881347),
            ("bf16", 4, 128, 2048, 0.564151, 0.729378, 0.046302, 0.385351, 0.557262, 2.572715),
        ),
    },
    "mc2-rack08-r1": {
        "label": "mc2-rack08-r1", "kind": "mc2_chain", "status": "completed", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "stack": "rack08", "date": "2026-10-21", "dtype": "bf16",
        "observed_expansion_mode": "not_observed", "requested_expansion_mode": "unspecified",
        "placement": "intra-node-ub", "nodes": ("rack08",),
        "source": "capture://mindie-sim-data/a5/2026-10-21/mc2/a5mc2-rack08-r1",
        "chains": (
            ("bf16", 2, 64, 512, 0.725649, 0.640399, 0.060283, 0.339579, 0.736450, 2.450751),
            ("bf16", 2, 64, 1024, 1.169757, 0.735081, 0.059138, 0.382390, 1.170900, 3.485126),
            ("bf16", 2, 128, 512, 0.725519, 1.139612, 0.054658, 0.608666, 0.733465, 3.291353),
            ("bf16", 2, 128, 1024, 1.159727, 1.272067, 0.058294, 0.657917, 1.172485, 4.376257),
            ("bf16", 4, 64, 1024, 0.837001, 0.374424, 0.060063, 0.206267, 0.826956, 2.304515),
            ("bf16", 4, 64, 2048, 1.081457, 0.862884, 0.057125, 0.445065, 1.054166, 3.299521),
            ("bf16", 4, 128, 1024, 0.782941, 0.644155, 0.057070, 0.339903, 0.826301, 2.600591),
            ("bf16", 4, 128, 2048, 1.074313, 0.733797, 0.057580, 0.387100, 1.066279, 3.243486),
        ),
    },
    "mc2-quant": {
        "label": "mc2-quant", "kind": "mc2_chain", "status": "completed", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "stack": "rack06", "date": "2026-10-22", "dtype": "int8+fp8-e4m3",
        "observed_expansion_mode": "not_observed", "requested_expansion_mode": "unspecified",
        "source": "capture://mindie-sim-data/a5/2026-10-22/mc2-quant/results/a5-mc2-quant-chain-validation.json",
        "all_to_allv_series": (
            ("fp8-e4m3", 2, ((14680064, 0.697195), (29360128, 1.161461))),
            ("fp8-e4m3", 4, ((11010048, 0.580367), (22020096, 0.680918), (44040192, 0.912123))),
            ("fp8-e4m3", 8, ((6422528, 0.751798), (12845056, 0.816551), (25690112, 0.865811))),
            ("int8", 2, ((7340032, 0.482216), (14680064, 0.767949))),
            ("int8", 4, ((5505024, 0.518167), (11010048, 0.577394), (22020096, 0.718820))),
            ("int8", 8, ((3211264, 0.892829), (6422528, 0.907955), (12845056, 0.953864))),
        ),
        "chains": (
            ("fp8-e4m3", 2, 64, 512, 0.698115, 0.381094, 0.065935, 0.222893, 0.732333, 1.810276),
            ("fp8-e4m3", 2, 64, 1024, 1.142767, 0.455798, 0.068328, 0.251140, 1.190490, 2.892251),
            ("fp8-e4m3", 2, 128, 512, 0.688892, 0.664670, 0.063380, 0.380689, 0.714672, 2.222401),
            ("fp8-e4m3", 2, 128, 1024, 1.110579, 0.726918, 0.064451, 0.412010, 1.190908, 3.304729),
            ("fp8-e4m3", 2, 256, 512, 0.653733, 1.231870, 0.066578, 0.691388, 0.699569, 3.100795),
            ("fp8-e4m3", 2, 256, 1024, 1.110908, 1.303427, 0.068515, 0.728705, 1.200703, 4.223019),
            ("fp8-e4m3", 4, 64, 512, 0.585642, 0.211950, 0.086684, 0.130867, 0.606183, 1.104670),
            ("fp8-e4m3", 4, 64, 1024, 0.687066, 0.249990, 0.087523, 0.147743, 0.709048, 1.378151),
            ("fp8-e4m3", 4, 64, 2048, 0.926047, 0.459246, 0.083581, 0.256325, 0.950041, 2.235708),
            ("fp8-e4m3", 4, 128, 512, 0.577523, 0.350969, 0.080985, 0.208733, 0.594604, 1.302188),
            ("fp8-e4m3", 4, 128, 1024, 0.691216, 0.381873, 0.081918, 0.223904, 0.711135, 1.628213),
            ("fp8-e4m3", 4, 128, 2048, 0.918893, 0.457738, 0.078873, 0.251819, 0.933813, 2.305786),
            ("fp8-e4m3", 4, 256, 512, 0.590566, 0.634466, 0.080767, 0.364188, 0.602926, 1.743749),
            ("fp8-e4m3", 4, 256, 1024, 0.692578, 0.668150, 0.080866, 0.384252, 0.712179, 2.074702),
            ("fp8-e4m3", 4, 256, 2048, 0.914441, 0.728394, 0.078618, 0.413897, 0.961075, 2.750947),
            ("fp8-e4m3", 8, 64, 512, 0.709558, 0.122965, 0.070783, 0.069817, 0.729712, 1.332910),
            ("fp8-e4m3", 8, 64, 1024, 0.813652, 0.139217, 0.071118, 0.077221, 0.822352, 1.552279),
            ("fp8-e4m3", 8, 64, 2048, 0.865880, 0.262502, 0.067163, 0.151531, 0.885621, 1.612864),
            ("fp8-e4m3", 8, 128, 512, 0.769478, 0.190995, 0.068700, 0.118271, 0.786279, 1.525838),
            ("fp8-e4m3", 8, 128, 1024, 0.980254, 0.243105, 0.267405, 0.250401, 0.994929, 1.745390),
            ("fp8-e4m3", 8, 128, 2048, 0.860041, 0.243084, 0.065247, 0.138976, 0.880412, 1.633653),
            ("fp8-e4m3", 8, 256, 512, 0.784726, 0.331123, 0.063708, 0.194159, 0.857284, 1.668988),
            ("fp8-e4m3", 8, 256, 1024, 0.873403, 0.347715, 0.063948, 0.203662, 0.892132, 1.697678),
            ("fp8-e4m3", 8, 256, 2048, 0.939583, 0.379220, 0.064436, 0.220560, 0.956580, 1.813527),
            ("int8", 2, 64, 512, 0.445207, 0.382156, 0.083731, 0.227210, 0.552574, 1.419045),
            ("int8", 2, 64, 1024, 0.681747, 0.454641, 0.079432, 0.252455, 0.860844, 2.117669),
            ("int8", 2, 128, 512, 0.438164, 0.670262, 0.075173, 0.382641, 0.529854, 1.851327),
            ("int8", 2, 128, 1024, 0.693525, 0.727219, 0.073746, 0.414332, 0.854900, 2.575234),
            ("int8", 2, 256, 512, 0.436241, 1.239563, 0.075956, 0.697151, 0.545780, 2.732471),
            ("int8", 2, 256, 1024, 0.693022, 1.301931, 0.075388, 0.731219, 0.862654, 3.476723),
            ("int8", 4, 64, 512, 0.519590, 0.208967, 0.069071, 0.126235, 0.547370, 1.001934),
            ("int8", 4, 64, 1024, 0.570741, 0.247240, 0.085084, 0.144373, 0.610138, 1.174772),
            ("int8", 4, 64, 2048, 0.685872, 0.461145, 0.085208, 0.256691, 0.779092, 1.868567),
            ("int8", 4, 128, 512, 0.498649, 0.348477, 0.079360, 0.208073, 0.538128, 1.183780),
            ("int8", 4, 128, 1024, 0.574444, 0.381856, 0.081773, 0.223897, 0.622388, 1.418903),
            ("int8", 4, 128, 2048, 0.680665, 0.456643, 0.078420, 0.252303, 0.803055, 1.911862),
            ("int8", 4, 256, 512, 0.496450, 0.637439, 0.081825, 0.368931, 0.553683, 1.622204),
            ("int8", 4, 256, 1024, 0.560550, 0.665425, 0.078874, 0.382763, 0.627321, 1.866134),
            ("int8", 4, 256, 2048, 0.703575, 0.729464, 0.079383, 0.415088, 0.800659, 2.374957),
            ("int8", 8, 64, 512, 0.992178, 0.238795, 0.268901, 0.237902, 0.986625, 1.728623),
            ("int8", 8, 64, 1024, 0.909792, 0.142100, 0.077072, 0.079955, 0.946427, 1.723251),
            ("int8", 8, 64, 2048, 0.935160, 0.263884, 0.075673, 0.153133, 0.988990, 1.748934),
            ("int8", 8, 128, 512, 0.893496, 0.192917, 0.075840, 0.121480, 0.927431, 1.662336),
            ("int8", 8, 128, 1024, 0.917294, 0.208441, 0.076093, 0.127949, 0.956187, 1.711356),
            ("int8", 8, 128, 2048, 0.949957, 0.245321, 0.076111, 0.143399, 1.006362, 1.756700),
            ("int8", 8, 256, 512, 0.900586, 0.333428, 0.076816, 0.198652, 0.930746, 1.688712),
            ("int8", 8, 256, 1024, 0.901163, 0.350263, 0.075252, 0.207167, 0.946631, 1.713711),
            ("int8", 8, 256, 2048, 0.979099, 0.413856, 0.273557, 0.260324, 1.044057, 1.959423),
        ),
    },
}


import json
from typing import Mapping

from mindie_sim.schema.common import ModelingError

INTRA_NODE_MAX_RANKS = 8
NATIVE_WINDOW_WARMUP_CALLS = 3
NATIVE_WINDOW_ADMITTED_REPEATS = 5

_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_CANARY_OK = frozenset({
    "all_ranks_matched", "ok", "passed", "matched", "match", "completed",
})
_SCOPES = {
    "intra_node_ub": "intra_node_ub",
    "intra": "intra_node_ub",
    "intra_node": "intra_node_ub",
    "ub": "intra_node_ub",
    "cross_node_urma": "cross_node_urma",
    "cross_node": "cross_node_urma",
    "cross": "cross_node_urma",
    "urma": "cross_node_urma",
    "cross_node_urma_far": "cross_node_urma_far",
    "far": "cross_node_urma_far",
}
_PAIR_PARTS = {"0->1", "0->4", "0->8", "0->24"}



class _Resolution(dict):
    """Mapping whose resolved fields are also readable as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _normalized_text(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())


def _is_a5(hardware_key, arch_family=None):
    if arch_family is not None and str(arch_family).strip().upper() != A5_ARCH_FAMILY:
        return False
    key = str(hardware_key or "").strip()
    if key in {A5_HARDWARE_KEY, A5_SKU}:
        return True
    folded = key.lower()
    return (folded in {A5_HARDWARE_KEY.lower(), A5_SKU.lower(), "a5",
                       "ascend950dt"} or folded.startswith("ascend950dt_9582"))


def _require_a5(hardware_key, arch_family=None):
    if not _is_a5(hardware_key, arch_family):
        raise ModelingError("a5_calibration_rejects_non_a5_hardware")


def _mode(value):
    text = str(value or "").strip().lower()
    if text in {"", "unspecified", "not_observed", "default", "auto", "none"}:
        return "unspecified"
    if text in {"aicpu", "ai_cpu", "cpu"}:
        return "aicpu"
    if text in {"aiv", "ai_vector_core", "vector", "aivector"}:
        return "aiv"
    return text


def _observed_mode(value):
    text = str(value or "").strip().lower()
    return text if text in _EXPANSION_MODES else "not_observed"


def _profiler_modes(evidence):
    """Extract explicitly named expansion engines from profiler text."""
    modes, stack = set(), [evidence]
    while stack:
        if modes >= _EXPANSION_MODES:
            break
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, value in item.items():
                stack.extend((key, value))
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
            continue
        if item is None or item is True or item is False:
            continue
        text = _normalized_text(item)
        if "aicpu" in text:
            modes.add("aicpu")
        if "aiv" in text or "aivectorcore" in text:
            modes.add("aiv")
    return modes


def resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode, *,
                             profiler_evidence=None):
    """Resolve one expansion regime without promoting an unconfirmed request."""
    raw_req = _mode(requested_expansion_mode)
    observed = _observed_mode(observed_expansion_mode)
    requested = raw_req if raw_req in _EXPANSION_MODES else "unspecified"
    evidence = _profiler_modes(profiler_evidence)
    regime, conflict = "unspecified", False
    if observed in _EXPANSION_MODES:
        regime = observed
        conflict = bool(evidence and evidence != {observed})
    elif len(evidence) == 1:
        regime = next(iter(evidence))
    elif evidence:
        conflict = True
    confirmed = regime in _EXPANSION_MODES and not conflict
    result = {
        "requested_expansion_mode": requested,
        "observed_expansion_mode": observed,
        "regime": regime if confirmed else "unspecified",
        "expansion_unconfirmed": not confirmed,
        "profiler_confirmed": bool(evidence) and regime in evidence and not conflict,
        "profiler_modes": sorted(evidence),
        "profiler_evidence": profiler_evidence,
        "confirmed_by": ("observed" if observed in _EXPANSION_MODES and not conflict
                         else "profiler" if confirmed else None),
        "evidence_conflict": conflict,
        "requested_mismatch": (observed in _EXPANSION_MODES
                               and requested in _EXPANSION_MODES
                               and observed != requested),
    }
    result["unconfirmed_tag"] = (None if confirmed else
                                 f"requested_{requested}_unconfirmed"
                                 if requested in _EXPANSION_MODES else None)
    return _Resolution(result)


def _collective(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "allreduce": "all_reduce", "all_reduce_only": "all_reduce",
        "allgather": "all_gather", "all_gather_only": "all_gather",
        "reduce_scatter_only": "reduce_scatter", "reduce_scatter_v": "reduce_scatter",
        "alltoall": "all_to_all", "all_to_all_only": "all_to_all",
        "alltoallv": "all_to_allv", "all_to_allv_only": "all_to_allv",
        "p2p": "p2p_send_recv", "p2p_only": "p2p_send_recv",
        "send_recv": "p2p_send_recv", "point_to_point": "p2p_send_recv",
    }
    return aliases.get(text, text)


def _dtype(value):
    text = str(value or "bf16").strip().lower().replace("_", "-")
    if text in {"fp8", "fp8e4m3", "fp8-e4m3", "e4m3", "fp8-e4m3fn"}:
        return "fp8-e4m3"
    if text in {"int8", "w8a8", "s8"}:
        return "int8"
    if text in {"bf16", "bfloat16"}:
        return "bf16"
    return text


def _explicit_scope(value):
    text = str(value or "").strip().lower()
    if text in _SCOPES:
        return _SCOPES[text]
    for token, scope in (("far", "cross_node_urma_far"), ("intra", "intra_node_ub"),
                         ("cross", "cross_node_urma"), ("urma", "cross_node_urma"),
                         ("2node", "cross_node_urma"), ("4node", "cross_node_urma")):
        if token in text:
            return scope
    return None


def _scope(value, world_size):
    return _explicit_scope(value) or (
        "cross_node_urma" if int(world_size) > INTRA_NODE_MAX_RANKS
        else "intra_node_ub")


def _explicit_pair(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return f"{int(value[0])}->{int(value[1])}"
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        source, destination = value.get("source"), value.get("destination")
        if source is not None and destination is not None:
            return f"{int(source)}->{int(destination)}"
    text = str(value or "").strip().lower()
    if text in _PAIR_PARTS:
        return text
    compact = "".join(text.replace("->", ">").replace("-", ">").replace(":", ">")
                      .replace("_", ">").split())
    if ">" not in compact:
        return None
    left, right = compact.split(">", 1)
    return f"{left}->{right}" if left.isdigit() and right.isdigit() else None


def _pair(value, collective, scope, world_size):
    if collective != "p2p_send_recv":
        return None
    if explicit := _explicit_pair(value):
        return explicit
    text = str(value or "").strip().lower()
    if "far" in text or scope == "cross_node_urma_far":
        return "0->24"
    if scope == "intra_node_ub":
        return "0->1"
    if "cross" in text or int(world_size) > INTRA_NODE_MAX_RANKS:
        return "0->8"
    if "intra" in text:
        return "0->1"
    return "0->8" if scope.startswith("cross") else "0->1"


def _pair_scope(pair, world_size):
    if pair == "0->1":
        return "intra_node_ub"
    if pair == "0->24":
        return "cross_node_urma_far"
    return ("cross_node_urma"
            if pair and int(world_size) > INTRA_NODE_MAX_RANKS else None)


def _split(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"train", "heldout"}:
        return text
    raise ValueError("service_sample_split_requires_train_or_heldout")


def _protocol_key(value):
    if value is None:
        return None
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            if isinstance(value, Mapping) else str(value))


def _same_canary(expected, observed):
    if expected is None or observed is None:
        return False
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        return all(key in observed and _same_canary(value, observed[key])
                   for key, value in expected.items())
    if (isinstance(expected, (int, float)) and isinstance(observed, (int, float))
            and not isinstance(expected, bool) and not isinstance(observed, bool)):
        return float(expected) == float(observed)
    return expected == observed or str(expected) == str(observed)


def _canary_issue(value):
    if value is None or value is True:
        return None
    if value is False:
        return "canary_failed"
    if isinstance(value, str):
        return None if value.strip().lower() in _CANARY_OK else f"canary={value!r}"
    if isinstance(value, Mapping):
        if "expected" not in value or "observed" not in value:
            return f"canary_incomplete={value!r}"
        expected, observed = value.get("expected"), value.get("observed")
        if expected is None or observed is None or not _same_canary(expected, observed):
            return f"canary={expected!r}!={observed!r}"
        return None
    return f"canary_invalid={value!r}"


def _status_issue(document):
    for key in ("status", "state"):
        if key in document and str(document[key]).strip().lower() not in {"completed", "ok"}:
            return f"{key}={document[key]!r}"
    if any(str(status).strip().lower() not in {"completed", "ok"}
           for status in document.get("rank_status") or ()):
        return f"rank_status={document['rank_status']!r}"
    for key in ("runs", "rank_reports", "cases"):
        children = document.get(key) or ()
        for child in children if isinstance(children, (list, tuple)) else ():
            if isinstance(child, Mapping) and (issue := _status_issue(child)):
                return issue
    return None


def _validate_row(document, label):
    if issue := _status_issue(document):
        raise ValueError(f"a5_campaign_not_completed:{label}:{issue}")
    for key in ("canary_ok", "canary_passed"):
        if key in document and document[key] is not True:
            raise ValueError(f"a5_campaign_canary_failed:{label}:{key}={document[key]!r}")
    for key in ("canary", "correctness"):
        issue = _canary_issue(document.get(key))
        if issue:
            raise ValueError(f"a5_campaign_canary_failed:{label}:{key}:{issue}")
    for target in (document, document.get("machine"), document.get("hardware"),
                   document.get("stack_binding")):
        if isinstance(target, Mapping) and any(
            key in target for key in ("hardware_key", "sku", "arch_family",
                                      "key", "architecture")
        ):
            _require_a5(
                target.get("hardware_key") or target.get("sku")
                or target.get("key") or A5_HARDWARE_KEY,
                target.get("arch_family") or target.get("architecture"))


_validate_campaign = _validate_row


def _selectors(document, row=None):
    row = row or {}
    row_proto, doc_proto = row.get("protocol"), document.get("protocol")
    return {
        "stack": row.get("stack", document.get("stack")),
        "cann": (row.get("cann") or (row.get("machine") or {}).get("cann")
                 or (row.get("stack_binding") or {}).get("cann")
                 or document.get("cann")
                 or (document.get("machine") or {}).get("cann")
                 or (document.get("stack_binding") or {}).get("cann")),
        "algorithm": (row.get("algorithm")
                      or (row_proto.get("algorithm")
                          if isinstance(row_proto, Mapping) else None)
                      or document.get("algorithm")
                      or (doc_proto.get("algorithm")
                          if isinstance(doc_proto, Mapping) else None)),
        "protocol": row_proto if "protocol" in row else doc_proto,
    }


def _merge_obs(parent, child):
    merged = {**parent, **child}
    if "observed_expansion_mode" in child or "observed_execution" in child:
        merged["observed_expansion_mode"] = child.get(
            "observed_expansion_mode", child.get("observed_execution"))
    sel = _selectors(child)
    for key in ("stack", "cann", "algorithm"):
        if sel[key] is not None:
            merged[key] = sel[key]
    return merged


def _canonical_semantic(value):
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _fallback_measurement(collective, boundary, algorithm_identified=False):
    """Explicit campaign fallback; never claims a native AI_CPU child."""
    if boundary == "resource_phase":
        return {
            "kind": "resource_phase", "phase": "collective_phase",
            "resource": "collective_phase",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_resource_phase",
            "service_semantics": "collective_phase_service",
            "complete_collective_call": False, "complete_directed_call": False,
            "link_transit_identified": False, "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": False, "explicit": False,
        }
    if collective == "p2p_send_recv":
        return {
            "kind": "device_call", "phase": "directed_p2p_call",
            "resource": "directed_p2p_call",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_device_call",
            "service_semantics": "directed_p2p_call",
            "complete_collective_call": False, "complete_directed_call": True,
            "link_transit_identified": False, "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": True, "explicit": False,
        }
    return {
        "kind": "device_call", "phase": "collective_call",
        "resource": "collective_call",
        "work_coordinate": "captured_per_rank_payload",
        "duration_unit": "ms", "source_kind": "campaign_device_call",
        "service_semantics": "whole_collective_call_envelope",
        "complete_collective_call": True, "complete_directed_call": False,
        "link_transit_identified": False, "hop_scaling": "not_derived",
        "algorithm_identified": bool(algorithm_identified),
        "direction_observed": False, "explicit": False,
    }


def _validate_measurement_window(measurement, window):
    phase = measurement.get("phase")
    kind = window.get("kind")
    if phase == "aicpu_child" and kind != "native_single_target_call":
        raise ValueError("a5_measurement_transport_window_mismatch")
    if phase in {"collective_call", "directed_p2p_call", "collective_phase"} \
            and kind == "native_single_target_call":
        raise ValueError("a5_measurement_transport_window_mismatch")


def _fallback_transport_window(measurement):
    if measurement.get("phase") == "aicpu_child":
        return {
            "kind": "native_single_target_call", "phase": "aicpu_child",
            "warmup_calls": NATIVE_WINDOW_WARMUP_CALLS,
            "admitted_repeats": NATIVE_WINDOW_ADMITTED_REPEATS,
            "profiled_ranks": "0,group_size-1",
            "clock_domain": "device", "concurrent_collectives": False,
        }
    return {
        "kind": "campaign_steady_state", "phase": "per_op",
        "warmup_calls": None, "admitted_repeats": None,
        "clock_domain": "device", "concurrent_collectives": False,
    }
