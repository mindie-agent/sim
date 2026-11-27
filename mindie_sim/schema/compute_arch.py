"""Declared representation, native-compute, cast, runtime and data-path capability.

[Role] Normalize owner-local issue/service capacity and the separate capability
  facts a physical device declares: payload representations, native compute
  modes, legal casts, directional local data paths and runtime implementations.
[Invariants] A source API call is not an ISA instruction. Storing or transferring
  a format, casting it, computing it natively and having a current software
  implementation are four different capabilities. A compute rate never enables a
  native mode; a native mode names its operands, accumulator and execution unit.
  Paths are directional and keep their representation role, owner reach, target
  buffer and synchronization constraint. Missing declarations stay missing.
[Timeouts & Locks] Bounded input metadata; no waits or locks.
[Side-Effect Policy] Immutable declarations; no measurements, drivers or I/O.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math

from .common import ModelingError
from .compute_arch_presets import CAPABILITY_PRESETS
from .compute_peak_aliases import canonical_format
from .device_input import HBM_CACHELINE_SERVICE


REPRESENTATION_ROLES = frozenset({"storage", "transfer", "operand", "accumulator"})
NATIVE_COMPUTE_KINDS = frozenset({"matrix", "vector", "scalar"})
PATH_SYNCS = frozenset({"pipe", "event", "barrier", "none"})
STORAGE_DOMAINS = frozenset({
    "gm", "hbm", "l1", "l0a", "l0b", "l0c", "ub", "ssbuffer", "reg",
})

_CAPABILITY_MAP_FIELDS = (
    "instruction_services", "representations", "native_compute", "casts",
    "paths", "runtimes",
)


def _text(value, reason, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ModelingError(f"{reason}:{value}")
    return value


def _positive_int(value, reason):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{reason}:{value}")
    return value


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

    @property
    def has_linked_supply(self) -> bool:
        return bool(self.scale_with or self.clock_hz is not None or self.name == HBM_CACHELINE_SERVICE)

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
class RepresentationCapability:
    """One payload format and the roles that format may legally fill.

    ``storage``/``transfer`` are byte-payload roles. ``operand`` means a native
    compute mode consumes it and ``accumulator`` means a native mode produces it.
    An FP8 payload that may be stored and copied is therefore not thereby a
    native FP8 matrix operand.
    """

    name: str
    bits: int
    roles: tuple[str, ...]
    alignment_bytes: int = 1
    provenance: object = field(default_factory=dict)

    def supports(self, role: str) -> bool:
        return role in self.roles

    def to_dict(self):
        return {"bits": self.bits, "roles": list(self.roles),
                "alignment_bytes": self.alignment_bytes,
                "provenance": self.provenance}


@dataclass(frozen=True)
class NativeComputeMode:
    """A native operand/accumulator pair on one declared execution unit."""

    name: str
    kind: str
    operands: tuple[str, ...]
    unit: str
    accumulators: tuple[str, ...] = ()
    implementations: tuple[str, ...] = ()
    provenance: object = field(default_factory=dict)

    def accepts(self, operand: str, accumulator: str | None = None) -> bool:
        if canonical_format(operand) not in self.operands and operand not in self.operands:
            return False
        if accumulator is None:
            return True
        return canonical_format(accumulator) in self.accumulators

    def to_dict(self):
        return {"kind": self.kind, "operands": list(self.operands),
                "unit": self.unit,
                **({"accumulators": list(self.accumulators)}
                   if self.accumulators else {}),
                **({"implementations": list(self.implementations)}
                   if self.implementations else {}),
                "provenance": self.provenance}


@dataclass(frozen=True)
class CastCapability:
    """A declared representation conversion (not a native compute mode)."""

    source: str
    target: str
    implementation: str = ""
    lossless: bool = False
    provenance: object = field(default_factory=dict)

    def to_dict(self):
        return {"source": self.source, "target": self.target,
                **({"implementation": self.implementation}
                   if self.implementation else {}),
                "lossless": self.lossless, "provenance": self.provenance}


@dataclass(frozen=True)
class DataPath:
    """A directional route with representation, owner reach, buffer and sync.

    ``unit`` names the transfer resource when the source declares one. It stays
    empty when the route's existence is sourced but its service role is not:
    the path is still queryable and never invents a rate.
    """

    name: str
    source: str
    target: str
    roles: tuple[str, ...]
    unit: str = ""
    executor: str = ""
    reach: tuple[str, ...] = ()
    buffer: str = ""
    sync: str = "pipe"
    alignment_bytes: int = 1
    representations: tuple[str, ...] = ()
    provenance: object = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return f"{self.source}->{self.target}"

    def supports(self, representation: str,
                 registry: dict[str, RepresentationCapability]) -> bool:
        name = canonical_format(representation)
        if self.representations and name not in self.representations:
            return False
        declared = registry.get(name) or registry.get(representation)
        if declared is None:
            return False
        return bool(set(self.roles) & set(declared.roles))

    def to_dict(self):
        return {"source": self.source, "target": self.target,
                "roles": list(self.roles),
                **({"unit": self.unit} if self.unit else {}),
                **({"executor": self.executor} if self.executor else {}),
                **({"reach": list(self.reach)} if self.reach else {}),
                **({"buffer": self.buffer} if self.buffer else {}),
                "sync": self.sync, "alignment_bytes": self.alignment_bytes,
                **({"representations": list(self.representations)}
                   if self.representations else {}),
                "provenance": self.provenance}


@dataclass(frozen=True)
class RuntimeCapability:
    """A concrete software implementation set, separate from architecture."""

    name: str
    version: str
    provides: tuple[str, ...] = ()
    conditions: tuple[tuple[str, str], ...] = ()
    provenance: object = field(default_factory=dict)

    def implements(self, implementation: str) -> bool:
        return implementation in self.provides

    def to_dict(self):
        return {"name": self.name, "version": self.version,
                **({"provides": list(self.provides)} if self.provides else {}),
                **({"conditions": dict(self.conditions)} if self.conditions else {}),
                "provenance": self.provenance}


@dataclass(frozen=True)
class ComputeArchSpec:
    instruction_services: tuple[InstructionService, ...] = ()
    preset: str = ""
    representations: tuple[RepresentationCapability, ...] = ()
    native_modes: tuple[NativeComputeMode, ...] = ()
    casts: tuple[CastCapability, ...] = ()
    paths: tuple[DataPath, ...] = ()
    runtimes: tuple[RuntimeCapability, ...] = ()
    clock_hz: float | None = None

    # ---- capability queries ------------------------------------------------
    @property
    def declared(self) -> bool:
        return bool(self.preset or self.clock_hz or self.instruction_services or self.has_capabilities())

    def has_capabilities(self) -> bool:
        return bool(self.representations or self.native_modes or self.casts
                    or self.paths or self.runtimes)

    def has_native_modes(self) -> bool:
        return bool(self.native_modes)

    def representation(self, name: str) -> RepresentationCapability:
        wanted = canonical_format(name)
        for item in self.representations:
            if item.name == wanted:
                return item
        raise ModelingError(f"representation_undeclared:{name}")

    def representation_supports(self, name: str, role: str) -> bool:
        if role not in REPRESENTATION_ROLES:
            raise ModelingError(f"representation_role_unknown:{role}")
        try:
            return self.representation(name).supports(role)
        except ModelingError:
            return False

    def native_modes_for(self, kind: str, operand: str,
                         accumulator: str | None = None
                         ) -> tuple[NativeComputeMode, ...]:
        """All declared native modes for one operand, not one forced API."""
        if kind not in NATIVE_COMPUTE_KINDS:
            raise ModelingError(f"native_compute_kind_unknown:{kind}")
        return tuple(mode for mode in self.native_modes
                     if mode.kind == kind and mode.accepts(operand, accumulator))

    def native_mode(self, kind: str, operand: str,
                    accumulator: str | None = None) -> NativeComputeMode:
        candidates = self.native_modes_for(kind, operand, accumulator)
        if not candidates:
            detail = f"{kind}:{operand}"
            if accumulator is not None:
                detail += f":acc={accumulator}"
            raise ModelingError(f"native_compute_mode_unsupported:{detail}")
        return candidates[0]

    def supports_native(self, kind: str, operand: str,
                        accumulator: str | None = None) -> bool:
        try:
            self.native_mode(kind, operand, accumulator)
            return True
        except ModelingError:
            return False

    def implemented_native_mode(self, kind: str, operand: str,
                                accumulator: str | None,
                                runtime: RuntimeCapability) -> NativeComputeMode:
        """Resolve an implemented candidate, not merely the first declaration.

        A semantic operand can have multiple native candidates. A runtime may
        implement a later candidate even when the first declared candidate is
        unavailable; only a candidate whose implementation id is provided by
        ``runtime`` is selected.
        """
        candidates = self.native_modes_for(kind, operand, accumulator)
        if not candidates:
            detail = f"{kind}:{operand}"
            if accumulator is not None:
                detail += f":acc={accumulator}"
            raise ModelingError(f"native_compute_mode_unsupported:{detail}")
        implemented = [
            mode for mode in candidates
            if mode.implementations
            and any(runtime.implements(item) for item in mode.implementations)
        ]
        if implemented:
            return implemented[0]
        undeclared = [mode.name for mode in candidates if not mode.implementations]
        if undeclared:
            raise ModelingError(
                f"native_mode_implementation_undeclared:{undeclared[0]}")
        raise ModelingError(
            "native_mode_not_implemented:"
            + ",".join(mode.name for mode in candidates)
            + ":" + runtime.name)

    def cast_rule(self, source: str, target: str) -> CastCapability | None:
        wanted = (canonical_format(source), canonical_format(target))
        for item in self.casts:
            if (item.source, item.target) == wanted:
                return item
        return None

    def can_cast(self, source: str, target: str) -> bool:
        if canonical_format(source) == canonical_format(target):
            try:
                self.representation(source)
                return True
            except ModelingError:
                return False
        return self.cast_rule(source, target) is not None

    def path(self, source: str, target: str, representation: str | None = None,
             *, executor: str | None = None) -> DataPath | None:
        registry = {item.name: item for item in self.representations}
        for item in self.paths:
            if item.source != source or item.target != target:
                continue
            if executor is not None and item.executor != executor:
                continue
            if representation is None or item.supports(representation, registry):
                return item
        return None

    def require_path(self, source: str, target: str,
                     representation: str | None = None, *,
                     executor: str | None = None) -> DataPath:
        found = self.path(source, target, representation, executor=executor)
        if found is None:
            detail = f"{source}->{target}"
            if representation is not None:
                detail += f":{representation}"
            raise ModelingError(f"data_path_unsupported:{detail}")
        return found

    def paths_from(self, source: str,
                   representation: str | None = None) -> tuple[DataPath, ...]:
        registry = {item.name: item for item in self.representations}
        return tuple(item for item in self.paths if item.source == source
                     and (representation is None
                          or item.supports(representation, registry)))

    def runtime(self, name: str) -> RuntimeCapability:
        for item in self.runtimes:
            if item.name == name:
                return item
        raise ModelingError(f"runtime_undeclared:{name}")

    def validate_peaks(self, peaks, *, aliases=()) -> None:
        """Native peaks need a declared native mode; aliases are removed first."""
        from .compute_peak_aliases import (
            native_compute_peak_items, validate_compute_peak_aliases)

        validate_compute_peak_aliases(peaks, aliases, self)
        _validate_peak_formats(dict(native_compute_peak_items(peaks, aliases)),
                               self.native_modes)

    # ---- parsing -----------------------------------------------------------
    @classmethod
    def from_mapping(cls, raw, *, source: str = "compute_arch"):
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ModelingError(f"{source}_mapping_required")
        declared = dict(raw)
        preset_name = declared.pop("preset", "")
        if preset_name:
            if preset_name not in CAPABILITY_PRESETS:
                raise ModelingError(f"compute_arch_preset_unknown:{preset_name}")
            # The preset is an immutable module constant; the parser only
            # reads it and builds fresh mappings, so a shallow projection is
            # enough and avoids copying the whole capability tree per parse.
            base = CAPABILITY_PRESETS[preset_name]
            effective = {**base, **declared}
            for key in _CAPABILITY_MAP_FIELDS:
                if key in base and key in declared:
                    effective[key] = {**base[key], **declared[key]}
            preset_name = str(preset_name)
        else:
            effective = declared
        raw_clock = effective.get("clock_hz")
        clock_hz: float | None = None
        if raw_clock is not None:
            if isinstance(raw_clock, bool) or not isinstance(raw_clock, (int, float)):
                raise ModelingError(f"compute_arch_clock_hz_invalid:{raw_clock}")
            clock_hz = float(raw_clock)
            if not math.isfinite(clock_hz) or clock_hz <= 0.0:
                raise ModelingError(f"compute_arch_clock_hz_invalid:{raw_clock}")
        services = _parse_instruction_services(effective.get("instruction_services", {}))
        representations = _parse_representations(effective.get("representations", {}))
        native_modes = _parse_native_modes(effective.get("native_compute", {}))
        casts = _parse_casts(effective.get("casts", {}))
        paths = _parse_paths(effective.get("paths", {}))
        runtimes = _parse_runtimes(effective.get("runtimes", {}))
        spec = cls(
            instruction_services=tuple(sorted(services, key=lambda item: item.name)),
            preset=preset_name,
            representations=tuple(sorted(representations, key=lambda item: item.name)),
            native_modes=tuple(sorted(native_modes, key=lambda item: item.name)),
            casts=tuple(sorted(casts, key=lambda item: (item.source, item.target))),
            paths=tuple(sorted(paths, key=lambda item: (item.source, item.target,
                                                        item.name))),
            runtimes=tuple(sorted(runtimes, key=lambda item: item.name)),
            clock_hz=clock_hz,
        )
        spec._validate_declarations()
        return spec

    def _validate_declarations(self):
        registry = {item.name: item for item in self.representations}
        for mode in self.native_modes:
            if mode.kind not in NATIVE_COMPUTE_KINDS:
                raise ModelingError(f"native_compute_kind_unknown:{mode.name}:{mode.kind}")
            for name in mode.operands:
                representation = registry.get(name)
                if representation is None:
                    raise ModelingError(
                        f"native_compute_representation_missing:{mode.name}:{name}")
                if not representation.supports("operand"):
                    raise ModelingError(
                        f"native_compute_operand_role_missing:{mode.name}:{name}")
            for name in mode.accumulators:
                representation = registry.get(name)
                if representation is None:
                    raise ModelingError(
                        f"native_compute_representation_missing:{mode.name}:{name}")
                if not representation.supports("accumulator"):
                    raise ModelingError(
                        f"native_compute_accumulator_role_missing:{mode.name}:{name}")
        for item in self.paths:
            if item.source not in STORAGE_DOMAINS or item.target not in STORAGE_DOMAINS:
                raise ModelingError(f"data_path_domain_unknown:{item.name}")
            if item.source == item.target:
                raise ModelingError(f"data_path_requires_distinct_domains:{item.name}")
            if not item.roles or not set(item.roles) <= REPRESENTATION_ROLES:
                raise ModelingError(f"data_path_roles_invalid:{item.name}")
            if item.sync not in PATH_SYNCS:
                raise ModelingError(f"data_path_sync_unknown:{item.name}:{item.sync}")
            for name in item.representations:
                if name not in registry:
                    raise ModelingError(
                        f"data_path_representation_missing:{item.name}:{name}")
                if not item.supports(name, registry):
                    raise ModelingError(f"data_path_representation_role_mismatch:{item.name}:{name}")
        for item in self.casts:
            if item.source not in registry or item.target not in registry:
                raise ModelingError(
                    f"cast_representation_missing:{item.source}->{item.target}")

    # ---- supply ------------------------------------------------------------
    def supplies(self, compute_units, resource_units, physical_rates=None):
        """Device totals retain per-owner partitioning; explicit rates override."""
        rates, units, work_units, modes = {}, {}, {}, {}
        # Resource owners must exist before a mode claims to share them,
        # regardless of alphabetical key order (e.g. get_sff... sharing
        # pipe_s_issue).
        services = tuple(sorted(self.instruction_services,
                                key=lambda item: (item.resource is not None, item.name)))
        for service in services:
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
                resource_count = units.get(
                    service.resource,
                    resource_units.get(service.resource, compute_units))
                if resource_count != count:
                    raise ModelingError("instruction_service_owner_mismatch:" + service.name)
                modes[service.name] = {"resource": service.resource,
                                       "rate": rate * count}
            else:
                rates[service.name] = rate * count
                units[service.name] = count
            work_units[service.name] = service.unit
        return rates, units, work_units, modes

    def to_dict(self):
        """Emit the resolved declaration, not only a preset reference.

        A serialized hardware document stays self-describing and reproducible
        even if the preset registry later gains a different declaration.
        """
        result = {}
        if self.preset:
            result["preset"] = self.preset
        if self.clock_hz is not None:
            result["clock_hz"] = self.clock_hz
        if self.instruction_services:
            result["instruction_services"] = {
                item.name: item.to_dict() for item in self.instruction_services}
        if self.representations:
            result["representations"] = {
                item.name: item.to_dict() for item in self.representations}
        if self.native_modes:
            result["native_compute"] = {
                item.name: item.to_dict() for item in self.native_modes}
        if self.casts:
            result["casts"] = {
                f"{item.source}->{item.target}": item.to_dict() for item in self.casts}
        if self.paths:
            result["paths"] = {item.name: item.to_dict() for item in self.paths}
        if self.runtimes:
            result["runtimes"] = {
                item.name: item.to_dict() for item in self.runtimes}
        return result


def _validate_peak_formats(peaks, modes) -> None:
    matrix = tuple(mode for mode in modes if mode.kind == "matrix")
    for dtype in peaks:
        aliases = {dtype, canonical_format(dtype)}
        covered = any(alias in mode.operands for mode in matrix for alias in aliases)
        if not covered:
            raise ModelingError(f"compute_peak_without_native_compute_mode:{dtype}")


def _parse_instruction_services(raw):
    if not isinstance(raw, Mapping):
        raise ModelingError("instruction_services_mapping_required")
    services = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise ModelingError(f"instruction_service_mapping_required:{name}")
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
    return services


def _parse_representations(raw):
    if not isinstance(raw, Mapping):
        raise ModelingError("representations_mapping_required")
    result = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise ModelingError(f"representation_mapping_required:{name}")
        roles = tuple(sorted(set(str(role) for role in item.get("roles", ()))))
        if not roles or not set(roles) <= REPRESENTATION_ROLES:
            raise ModelingError(f"representation_roles_invalid:{name}")
        result.append(RepresentationCapability(
            canonical_format(name),
            _positive_int(item.get("bits"), f"representation_bits_invalid:{name}"),
            roles,
            _positive_int(item.get("alignment_bytes", 1),
                          f"representation_alignment_invalid:{name}"),
            deepcopy(item.get("provenance", {})),
        ))
    return result


def _parse_native_modes(raw):
    if not isinstance(raw, Mapping):
        raise ModelingError("native_compute_mapping_required")
    result = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise ModelingError(f"native_compute_mapping_required:{name}")
        kind = _text(item.get("kind"), f"native_compute_kind_invalid:{name}")
        operands = tuple(sorted({canonical_format(value)
                                 for value in item.get("operands", ())}))
        accumulators = tuple(sorted({canonical_format(value)
                                     for value in item.get("accumulators", ())}))
        if not operands:
            raise ModelingError(f"native_compute_operands_required:{name}")
        result.append(NativeComputeMode(
            name, kind, operands,
            _text(item.get("unit"), f"native_compute_unit_invalid:{name}"),
            accumulators,
            tuple(sorted(set(str(value) for value in item.get("implementations", ())))),
            deepcopy(item.get("provenance", {})),
        ))
    return result


def _parse_casts(raw):
    if not isinstance(raw, Mapping):
        raise ModelingError("casts_mapping_required")
    result = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise ModelingError(f"cast_mapping_required:{name}")
        source = canonical_format(item.get("source", ""))
        target = canonical_format(item.get("target", ""))
        if not source or not target or source == target:
            raise ModelingError(f"cast_requires_distinct_representations:{name}")
        result.append(CastCapability(
            source, target,
            _text(item.get("implementation", ""),
                  f"cast_implementation_invalid:{name}", allow_empty=True),
            bool(item.get("lossless", False)),
            deepcopy(item.get("provenance", {})),
        ))
    return result


def _parse_paths(raw):
    if not isinstance(raw, Mapping):
        raise ModelingError("paths_mapping_required")
    result = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise ModelingError(f"path_mapping_required:{name}")
        roles = tuple(sorted(set(str(role) for role in item.get("roles",
                                                               ("storage", "transfer")))))
        result.append(DataPath(
            name,
            _text(item.get("source"), f"path_source_invalid:{name}"),
            _text(item.get("target"), f"path_target_invalid:{name}"),
            roles,
            _text(item.get("unit", ""), f"path_unit_invalid:{name}", allow_empty=True),
            _text(item.get("executor", ""), f"path_executor_invalid:{name}",
                  allow_empty=True),
            tuple(sorted(set(str(value) for value in item.get("reach", ())))),
            _text(item.get("buffer", ""), f"path_buffer_invalid:{name}", allow_empty=True),
            str(item.get("sync", "pipe")),
            _positive_int(item.get("alignment_bytes", 1),
                          f"path_alignment_invalid:{name}"),
            tuple(sorted({canonical_format(value)
                          for value in item.get("representations", ())})),
            deepcopy(item.get("provenance", {})),
        ))
    return result


def _parse_runtimes(raw):
    if not isinstance(raw, Mapping):
        raise ModelingError("runtimes_mapping_required")
    result = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise ModelingError(f"runtime_mapping_required:{name}")
        conditions = item.get("conditions", {})
        if not isinstance(conditions, Mapping):
            raise ModelingError(f"runtime_conditions_invalid:{name}")
        result.append(RuntimeCapability(
            _text(item.get("name", name), f"runtime_name_invalid:{name}"),
            _text(item.get("version"), f"runtime_version_invalid:{name}"),
            tuple(sorted(set(str(value) for value in item.get("provides", ())))),
            tuple(sorted((str(key), str(value))
                         for key, value in conditions.items())),
            deepcopy(item.get("provenance", {})),
        ))
    return result


__all__ = [
    "CAPABILITY_PRESETS",
    "NATIVE_COMPUTE_KINDS",
    "PATH_SYNCS",
    "REPRESENTATION_ROLES",
    "STORAGE_DOMAINS",
    "CastCapability",
    "ComputeArchSpec",
    "DataPath",
    "InstructionService",
    "NativeComputeMode",
    "RepresentationCapability",
    "RuntimeCapability",
]
