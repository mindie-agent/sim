"""Declared small-operation service on the existing physical resource engine.

[Role] Normalize owner-local issue/service capacity with explicit count units.
[Invariants] A source API call is not an ISA instruction. An effective issue
  interval is supply, not an extra dependency latency or observed kernel span.
  Missing services remain unknown; device peaks never invent their rates.
[Timeouts & Locks] Bounded input metadata; no waits or locks.
[Side-Effect Policy] Immutable declarations; no measurements, drivers or I/O.
"""
from dataclasses import dataclass
import math

from .common import ModelingError


@dataclass(frozen=True)
class InstructionService:
    name: str
    unit: str
    owner_axis: str
    per_owner_rate: float
    provenance: object
    scale_with: tuple[str, float] | None = None
    resource: str | None = None
    clock_hz: float | None = None
    cycles_per_unit: float | None = None

    def to_dict(self):
        supply = ({"clock_hz": self.clock_hz, "cycles_per_unit": self.cycles_per_unit}
                  if self.clock_hz is not None else {"per_owner_rate": self.per_owner_rate})
        return {"unit": self.unit, "owner_axis": self.owner_axis,
                **supply, "provenance": self.provenance,
                **({"resource": self.resource} if self.resource else {}),
                **({"scale_with": {"resource": self.scale_with[0],
                    "reference_per_owner_rate": self.scale_with[1]}}
                   if self.scale_with else {})}


@dataclass(frozen=True)
class ComputeArchSpec:
    instruction_services: tuple[InstructionService, ...] = ()

    @classmethod
    def from_mapping(cls, raw):
        services = []
        for name, item in raw.get("instruction_services", {}).items():
            has_rate, has_interval = "per_owner_rate" in item, "issue_ns" in item
            has_cycles = "cycles_per_unit" in item
            if sum((has_rate, has_interval, has_cycles)) != 1:
                raise ModelingError("instruction_service_requires_rate_or_interval:" + name)
            value = item["per_owner_rate"] if has_rate else item[
                "cycles_per_unit" if has_cycles else "issue_ns"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ModelingError("instruction_service_nonpositive_supply:" + name)
            unit, axis = item["unit"], item["owner_axis"]
            if not unit or not axis:
                raise ModelingError("instruction_service_requires_unit_and_owner:" + name)
            clock = item.get("clock_hz") if has_cycles else None
            if has_cycles and (isinstance(clock, bool) or not isinstance(clock, (int, float))
                    or not math.isfinite(clock) or clock <= 0):
                raise ModelingError("instruction_service_requires_positive_clock:" + name)
            scale = item.get("scale_with")
            if scale is not None:
                reference = scale["reference_per_owner_rate"]
                if (isinstance(reference, bool) or not isinstance(reference, (int, float))
                        or not math.isfinite(reference) or reference <= 0):
                    raise ModelingError("instruction_service_invalid_reference_supply:" + name)
                scale = (scale["resource"], float(reference))
            services.append(InstructionService(name, unit, axis,
                float(value) if has_rate else (clock if has_cycles else 1e9) / value,
                item.get("provenance", {}), scale, item.get("resource"),
                clock, float(value) if has_cycles else None))
        return cls(tuple(sorted(services, key=lambda item: item.name)))

    def supplies(self, compute_units, resource_units, physical_rates=None):
        """Device totals retain per-owner partitioning; explicit rates override."""
        rates, units, work_units, modes = {}, {}, {}, {}
        for service in self.instruction_services:
            count = resource_units.get(service.owner_axis, compute_units)
            rate = service.per_owner_rate
            if service.scale_with:
                role, reference = service.scale_with
                # This is an explicit transfer assumption, not a deduction from
                # PMU correlation. Divide aggregate supply by its own owners;
                # adding idle cores must not accelerate one pinned instruction.
                if role not in (physical_rates or {}):
                    raise ModelingError("instruction_service_scale_resource_missing:" + role)
                source_owners = resource_units.get(role, compute_units)
                rate *= physical_rates[role] / source_owners / reference
            if service.resource:
                # A source-owned pipe mode shares its physical queue and owner
                # capacity; it must not become a second independent resource.
                if resource_units.get(service.resource, compute_units) != count:
                    raise ModelingError("instruction_service_owner_mismatch:" + service.name)
                modes[service.name] = {"resource": service.resource, "rate": rate * count}
            else:
                rates[service.name] = rate * count
                units[service.name] = count
            work_units[service.name] = service.unit
        return rates, units, work_units, modes

    def to_dict(self):
        return {"instruction_services": {item.name: item.to_dict()
                                         for item in self.instruction_services}}
