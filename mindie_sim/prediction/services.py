"""Work-dependent primitive service on the common physical resource engine.

[Role] Interpolate a declared primitive response and constrain its service supply.
[Invariants] Categorical identity is fixed; only work is interpolated. Native
  work and shared capacity are unchanged. Startup is charged once per invocation.
[Timeouts & Locks] Bounded tables, binary search, no locks or I/O.
[Side-Effect Policy] Builds programs and records bounded calibration diagnostics.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .program import KernelProgram, ResourceEnvelope, Stage


def service_identity(case, op, scope="none", *, device=None):
    metadata = dict(op.metadata)
    device = device or case.machine.dies[
        op.stage * case.runtime.parallel.tp * case.runtime.parallel.pcp]
    identity = {
        "hardware": {
            "key": case.machine.hardware_key or case.machine.sku,
            "architecture": case.machine.arch_family,
            "compute_units": device.compute_units,
            "resource_units": dict(device.resource_units),
        },
        "runtime": case.runtime.vllm_impl,
        "revision": case.runtime.vllm_version,
        "format": case.runtime.kernel_format,
        "kernel": metadata.get("kernel", metadata.get("kernel_name", op.recipe)),
        "implementation": metadata.get("implementation", {}),
        "dtype": op.dtype or case.model.compute_dtype,
        "layout": op.layout, "replay_scope": scope or "none",
        "axes": metadata.get("service_axes", {}),
    }
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class ServiceEstimate:
    work: float
    startup_ms: float
    service_ms: float
    basis: str

    @property
    def duration_ms(self):
        return self.startup_ms + self.service_ms


@dataclass(frozen=True)
class ServiceCurve:
    resource: str
    work_unit: str
    identity: Mapping
    boundary: str
    active_units: int
    startup_ms: float
    knots: tuple[tuple[float, float], ...]
    provenance: Mapping = field(default_factory=dict)
    model_id: str = field(init=False)
    _coordinates: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.boundary not in {"device_kernel", "resource_phase"}:
            raise ModelingError("service_curve_requires_device_or_resource_phase_boundary")
        if self.active_units < 1:
            raise ModelingError("service_curve_requires_positive_active_units")
        if (len(self.knots) < 2 or not math.isfinite(self.startup_ms)
                or self.startup_ms < 0):
            raise ModelingError("service_curve_requires_two_knots_and_nonnegative_startup")
        xs = tuple(x for x, _ in self.knots)
        if (any(not math.isfinite(x) or x <= 0 for x in xs)
                or any(b <= a for a, b in zip(xs, xs[1:]))
                or any(not math.isfinite(t) or t <= self.startup_ms for _, t in self.knots)):
            raise ModelingError("service_curve_requires_ordered_work_and_positive_service")
        object.__setattr__(self, "identity", json.loads(json.dumps(self.identity, allow_nan=False)))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        object.__setattr__(self, "_coordinates", xs)
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "model_id", hashlib.sha256(payload.encode()).hexdigest())

    @classmethod
    def from_mapping(cls, data):
        if data["schema"] != "mindie-sim.service-curve/1":
            raise ModelingError("service_curve_schema_unknown")
        return cls(data["resource"], data["work_unit"], data["identity"], data["boundary"],
                   data["active_units"], data["startup_ms"], tuple(tuple(p) for p in data["knots"]),
                   data.get("provenance", {}))

    def to_dict(self):
        return {"schema": "mindie-sim.service-curve/1", "resource": self.resource,
                "work_unit": self.work_unit, "identity": deepcopy(self.identity),
                "boundary": self.boundary, "active_units": self.active_units,
                "startup_ms": self.startup_ms,
                "knots": self.knots, "provenance": deepcopy(self.provenance)}

    def estimate(self, work):
        """No extrapolation or categorical substitution is hidden in this lookup."""
        if not self._coordinates[0] <= work <= self._coordinates[-1]:
            return None
        right = bisect_left(self._coordinates, work)
        if self._coordinates[right] == work:
            duration, basis = self.knots[right][1], "anchor"
        else:
            x0, t0 = self.knots[right - 1]
            x1, t1 = self.knots[right]
            duration = t0 + (t1 - t0) * (work - x0) / (x1 - x0)
            basis = "interpolated"
        return ServiceEstimate(work, self.startup_ms, duration - self.startup_ms, basis)


@dataclass(frozen=True)
class ResourceService:
    """One isolated resource invocation; phase composition stays in its caller."""

    model: ServiceCurve | Mapping
    outside_domain: str = "analytic"
    units: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.model, ServiceCurve):
            object.__setattr__(self, "model", ServiceCurve.from_mapping(self.model))
        if self.outside_domain not in {"analytic", "error"}:
            raise ModelingError("service_domain_policy_requires_analytic_or_error")

    def lower(self, context):
        from .mechanism import Work
        curve = self.model
        if not context.work:
            return ResourceEnvelope().lower(context)
        if curve.identity != context.service_identity:
            raise ModelingError(f"service_identity_mismatch:{curve.model_id}")
        if curve.boundary != context.service_boundary:
            raise ModelingError("service_boundary_mismatch")
        role = curve.resource
        if curve.work_unit != context.work_units.get(role, "work"):
            raise ModelingError(f"service_work_unit_mismatch:{role}")
        if set(context.native_work) != {role}:
            raise ModelingError("service_curve_requires_one_explicit_resource_phase")
        amount = context.native_work.get(role, 0)
        if role == "cube":
            portions = dict(context.op.matrix_flops) or {curve.identity["dtype"]: amount}
            if len(portions) != 1 or next(iter(portions)) != curve.identity["dtype"]:
                raise ModelingError("service_curve_requires_one_compute_precision")
            work = Work(matrix=portions)
        else:
            work = Work(resources={role: amount})
        expected = context.bind(work)
        # One measured response cannot identify the occupancy of several pipes,
        # or several instruction modes sharing a pipe. The caller must split it.
        if len(expected) != 1 or expected != context.work:
            raise ModelingError("service_curve_requires_one_explicit_resource_phase")
        resource = next(iter(expected))
        owner_count = context.resource_units.get(context.physical_role(role), context.core_count)
        active_units = len(self.units) if self.units else owner_count
        if active_units != curve.active_units:
            raise ModelingError("service_active_units_mismatch")
        placement = {resource: tuple(self.units)} if self.units else {}
        estimate = curve.estimate(amount)
        if estimate is None and self.outside_domain == "error":
            raise ModelingError(f"service_work_outside_domain:{amount}")
        supply = context.work[resource] / estimate.service_ms if estimate else math.inf
        physical_supply = context.rates[resource] * active_units / owner_count
        if context.record_service is not None:
            context.record_service(
                curve, estimate.basis if estimate else "analytic_outside_domain",
                capacity_limited=estimate is not None and supply > physical_supply,
            )
        if estimate is None:
            return KernelProgram((
                Stage("setup", latency=context.setup_ms),
                Stage("service", context.work, after=("setup",), placement=placement),
            ), minimum_ms=context.minimum_ms)
        return KernelProgram((
            Stage("setup", latency=context.setup_ms + estimate.startup_ms),
            Stage("service", context.work, after=("setup",),
                  caps={resource: supply}, placement=placement),
        ), minimum_ms=context.minimum_ms)
