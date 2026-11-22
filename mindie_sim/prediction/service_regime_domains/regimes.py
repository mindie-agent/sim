"""Service regime selection, migration bindings and domain lookup.

[Role] Resolve calibrated service regimes and migration bindings for resource models.
[Invariants] Explicit provenance and positive rates.
[Timeouts & Locks] Pure in-memory lookup; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections.abc import Mapping
import json

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.response_functions import nonnegative as _nonnegative


MISSING = object()
MIGRATION_SCHEMA = "mindie-sim.regime-migration-readiness/1"
HANDOFF_SCHEMA = "mindie-sim.regime-resource-phase-handoff/1"
MIGRATION_STATUSES = frozenset({
    "resource_phase_ready", "slice_ready", "awaiting_binding_plan",
    "awaiting_resource_service", "awaiting_program_binding", "awaiting_conditions",
    "partial", "conflict", "link_rejected", "observation_replay"})
_PUBLIC_STATUS = {
    "ready": "resource_phase_ready", "slice_ready": "slice_ready", "partial": "partial",
    "awaiting_conditions": "awaiting_conditions", "conflict": "conflict",
    "awaiting_program_binding": "awaiting_program_binding",
    "no_binding": "awaiting_binding_plan", "not_eligible": "observation_replay",
    "observation_replay": "observation_replay"}
_IDENTIFIABILITY_KEYS = frozenset({
    "resource_identifiability", "limitations", "boundary", "parameter_boundary",
    "scaling_assumption", "response_semantics", "combination",
    "measurement_identity", "assignment", "data_roles"})
_NEGATIVE_IDENTIFIABILITY_MARKERS = (
    "not independent", "no independent", "not independently", "does not identify",
    "not identify", "no guessed", "not assigned to a guessed", "not a hardware",
    "not a bandwidth", "effective call response only", "engineered", "inclusive",
    "whole native", "whole-call", "whole call", "total duration", "not measured")


def _provenance_identifiability(provenance):
    """Preserve old evidence; its negative result does not gate a new service."""
    evidence = []
    for key in sorted(_IDENTIFIABILITY_KEYS):
        value = provenance.get(key) if isinstance(provenance, Mapping) else None
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(
            value, sort_keys=True, ensure_ascii=False, default=str)
        lowered = text.lower()
        marker = next((item for item in _NEGATIVE_IDENTIFIABILITY_MARKERS
                       if item in lowered), None)
        if marker is not None:
            evidence.append({"key": key, "marker": marker, "text": text})
    return {"negative": bool(evidence), "evidence": evidence}


def _split_links(links):
    """Separate explicit binding plans from old candidate-only lists."""
    if links is None:
        return {}, {}
    if not isinstance(links, Mapping):
        raise ModelingError("migration_links_require_mapping")
    plans, legacy = {}, {}
    for rule_id, value in links.items():
        if not isinstance(rule_id, str) or not rule_id:
            raise ModelingError("migration_link_requires_rule_id")
        if isinstance(value, Mapping):
            plans[rule_id] = value
            continue
        ids = (value,) if isinstance(value, str) else tuple(value) if isinstance(
            value, (list, tuple)) else None
        if ids is None or any(not isinstance(item, str) or not item for item in ids):
            raise ModelingError("migration_link_requires_service_id_sequence:" + rule_id)
        if len(set(ids)) != len(ids):
            raise ModelingError("migration_link_repeats_service:" + rule_id)
        legacy[rule_id] = ids
    return plans, legacy


def _service_field(service, name, default=MISSING):
    if isinstance(service, Mapping):
        return service.get(name, default)
    return getattr(service, name, default)


def normalize_service_handoff(service):
    """Read one already-normalized T06 handoff without re-normalizing it."""
    service_id = _service_field(service, "service_id")
    if service_id is MISSING:
        service_id = _service_field(service, "id")
    if not isinstance(service_id, str) or not service_id:
        raise ModelingError("migration_service_handoff_requires_id")
    resources = _service_field(service, "resources", ())
    if isinstance(resources, str):
        resources = (resources,)
    if not isinstance(resources, (list, tuple)):
        raise ModelingError("migration_service_handoff_requires_resources:" + service_id)
    conditions = _service_field(service, "conditions", {}) or {}
    identification = _service_field(service, "identification", {}) or {}
    provenance = _service_field(service, "provenance", {}) or {}
    if not all(isinstance(item, Mapping) for item in (conditions, identification, provenance)):
        raise ModelingError("migration_service_handoff_requires_mappings:" + service_id)
    native_boundary = (_service_field(service, "native_boundary", None)
                       or _service_field(service, "native_kernel", None)
                       or conditions.get("native_boundary")
                       or conditions.get("native_kernel") or "")
    return {
        "service_id": service_id,
        "scope": str(_service_field(service, "scope", "")),
        "resources": tuple(str(item) for item in resources),
        "phase": str(_service_field(service, "phase", "")),
        "work_unit": str(_service_field(service, "work_unit", "")),
        "boundary": str(_service_field(service, "boundary", "")),
        "fixed_us": _nonnegative(_service_field(service, "fixed_us", 0)),
        "effective_rate": _service_field(service, "effective_rate"),
        "aggregate_rate": _service_field(service, "aggregate_rate"),
        "per_owner_rate": _service_field(service, "per_owner_rate"),
        "rate_scope": str(_service_field(service, "rate_scope", "")),
        "active_units": _service_field(service, "active_units", 1),
        "conditions": dict(conditions),
        "identification": dict(identification),
        "provenance": dict(provenance),
        "native_boundary": str(native_boundary).strip(),
    }


def _target_authority(kind):
    return ("resource_rate_service" if kind == "independent_service"
            else "joint_phase_service")


def _no_plan_record(raw, decomposition, mode, old_evidence):
    rule_id = raw["id"]
    response_class = decomposition["response_class"]
    target, replacement_scope = "diagnostic_observation_replay", None
    blockers = list(decomposition.get("blockers", ()))
    if response_class == "resource_phase_supply":
        status, target = "resource_phase_ready", "resource_phase_supply"
        replacement_scope = "declared_physical_rates"
        blockers += ["t04_executor_must_bind_declared_supply_as_resource_work"]
        requires = ["T04"]
    elif old_evidence["negative"] and decomposition["decomposable"]:
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "provenance_declares_no_independent_resource_identifiability",
                     "t06_normalized_resource_service_required_for_switch"]
        requires = ["T06", "T04"]
    elif mode == "joint_native":
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "joint_native_total_duration_must_not_become_a_busy_ns_rate",
                     "t06_normalized_joint_or_resource_service_required_for_switch"]
        requires = ["T06", "T04"]
    elif mode == "mac_service" or response_class == "measured_issue_class":
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "measured_issue_class_is_not_a_shape_generalization",
                     "t06_normalized_shape_service_required_for_switch"]
        requires = ["T06", "T04"]
    elif decomposition["decomposable"]:
        status, target = "awaiting_resource_service", "resource_rate_service"
        blockers += ["t06_normalized_resource_service_not_linked"]
        requires = ["T06", "T04"]
    else:
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "whole_call_response_requires_explicit_observation_replay"]
        requires = []
    return {
        "rule_id": rule_id, "execution_mode": mode,
        "response_class": response_class,
        "production_authority": {
            "completion_floor": "legacy_completion_floor",
            "joint_native": "legacy_joint_native_busy_ns",
            "mac_service": "legacy_mac_service"}.get(mode, "legacy_" + mode),
        "target_authority": target, "status": status,
        "replacement_scope": replacement_scope,
        "decomposition": decomposition, "linked_services": [],
        "blockers": sorted(set(blockers)),
        "provenance_evidence": list(old_evidence["evidence"]),
        "coverage": None, "conditions": None, "diagnostics": None,
        "service_evidence": {}, "scope": None, "requires": requires}


from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.response_functions import (ResponseFunction, nonnegative as _nonnegative,
                                 normalize_migration_plan, provenance_roles,
                                 scaled_coordinate)
from mindie_sim.prediction.service_regime_domains import (evaluate_migration_binding,
                                     validate_service_regime_domains)


ROOTS = {"hardware", "runtime", "revision", "format", "kernel", "implementation",
         "dtype", "layout", "replay_scope", "recipe", "family", "phase",
         "local_shape", "semantic_shape", "padded_shape", "launch_shape",
         "metadata", "work", "resource_units", "core_count", "physical_rates", "transport"}
REDUCERS = {"sum": sum, "max": max, "min": min, "len": len,
            "product": math.prod, "positive_count": lambda x: sum(v > 0 for v in x)}
RULE_FIELDS = {"id", "match", "coordinates", "domain", "response", "provenance",
               "execution", "continuity"}
EXECUTION_FIELDS = {"kind", "resource", "reference_rate", "native_kernel"}
MAC_EXECUTION_FIELDS = {"kind", "resource", "native_kernel",
                        "tile_m", "tile_n", "tile_k", "slots"}


@dataclass(frozen=True)
class RegimeSelection:
    duration_ms: float
    execution: Mapping | None = None
    rule_id: str | None = None
    continuity_group: str | None = None
    response_basis: str | None = None
    coordinate_values: Mapping | None = None
    exterior_policy: str | None = None
    issued_work: float | None = None
    rate_flops_per_s: float | None = None
    active_owners: int | None = None
    stage: str | None = None
    #: The selected rule's declared provenance; physical admission reads it.
    provenance: Mapping | None = None

    @property
    def mode(self):
        """Normalized execution semantics, never an arbitrary mapping probe."""
        if self.execution is not None:
            return str(self.execution.get("kind"))
        return "completion_floor"

    def bind_joint(self, context):
        """Observed joint service occupies an explicitly declared device queue.

        No byte/cycle occupancy is identified by a complete kernel duration.
        The analytical schedule remains a reference, not simultaneous demand.
        """
        if self.mode != "joint_native":
            raise ModelingError("joint_native_service_requires_queue_and_native_boundary")
        execution = self.execution
        if (context.native_work.get("link") or context.op.family in {
                "collective", "alltoall", "allreduce", "allgather", "reduce_scatter"}
                or dict(context.op.metadata).get("device_kernel_group")):
            raise ModelingError("joint_native_service_requires_single_local_kernel")
        role = execution["resource"]
        if (role not in context.resource_names or context.work_units.get(role) != "busy_ns"
                or context.resource_units.get(role) != 1):
            raise ModelingError("joint_native_service_requires_declared_single_queue_busy_ns:" + role)
        name = context.resource_names[role]
        return {name: self.duration_ms * execution["reference_rate"] / 1000}

    def diagnostics(self):
        """Execution/response evidence recorded without conflating the two."""
        return {"rule_id": self.rule_id, "continuity_group": self.continuity_group,
                "execution_mode": self.mode, "response_basis": self.response_basis,
                "response_duration_ms": self.duration_ms,
                "issued_work": self.issued_work,
                "rate_flops_per_s": self.rate_flops_per_s,
                "active_owners": self.active_owners,
                "stage": self.stage,
                "coordinate_values": dict(self.coordinate_values or {}),
                "exterior_policy": self.exterior_policy}


def _path(path):
    parts = path.split(".")
    if parts[0] not in ROOTS or any(not p for p in parts):
        raise ModelingError("service_regime_requires_physical_axis:" + path)
    return parts


def lookup(axes, path):
    value = axes
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part, MISSING)
        elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def axis_equal(actual, declared):
    """JSON arrays and immutable source tuples name the same physical axis."""
    if isinstance(actual, (list, tuple)) and isinstance(declared, (list, tuple)):
        return len(actual) == len(declared) and all(
            axis_equal(a, b) for a, b in zip(actual, declared))
    if isinstance(actual, Mapping) and isinstance(declared, Mapping):
        return actual.keys() == declared.keys() and all(
            axis_equal(actual[key], declared[key]) for key in actual)
    return actual == declared


def coordinates(axes, declarations):
    result = {}
    for name, raw in declarations.items():
        value = lookup(axes, raw["path"])
        if value is MISSING:
            if "default" not in raw:
                return None
            value = raw["default"]
        if "reduce" in raw:
            if not isinstance(value, (list, tuple)) or not value:
                return None
            value = REDUCERS[raw["reduce"]](value)
        result[name] = scaled_coordinate(value, raw.get("scale", 1))
    return result


def _guard(value, bounds):
    low = bounds.get("min", 0)
    high = bounds.get("max", math.inf)
    low_ok = value >= low if bounds.get("min_inclusive", True) else value > low
    high_ok = value <= high if bounds.get("max_inclusive", True) else value < high
    return low_ok and high_ok


def _within(values, declared):
    return all(_guard(values[name], bounds) for name, bounds in declared.items())


def _guard_excluded(raw, values):
    """Whether a declared non-support guard excludes this physical point."""
    support = set((raw.get("continuity") or {}).get("support") or ())
    for name, bounds in raw.get("domain", {}).items():
        if name not in support and not _guard(values[name], bounds):
            return True
    return False


def _seam_owner(raw, key, values, matches, domains):
    """Mirror compiled half-open ownership for a shared closed endpoint."""
    continuity = raw.get("continuity")
    if continuity is None:
        return True
    declared = domains[key]
    for name, bounds in declared.items():
        high = bounds.get("max", math.inf)
        if values.get(name) != high or bounds.get("min", 0) == high:
            continue
        if not bounds.get("max_inclusive", True):
            continue
        for other, other_key, _result, other_values in matches:
            if other is raw:
                continue
            other_continuity = other.get("continuity") or {}
            if other_continuity.get("group") != continuity.get("group"):
                continue
            other_domain = domains[other_key]
            other_bounds = other_domain.get(name)
            if (other_bounds is None or other_bounds.get("min", 0) != high
                    or not other_bounds.get("min_inclusive", True)):
                continue
            if _within(other_values, other_domain):
                return False
    return True


def _execution(raw):
    """Omitted execution and an explicit completion floor are equivalent."""
    execution = raw.get("execution")
    if execution is None:
        return None
    if not isinstance(execution, Mapping):
        raise ModelingError("joint_native_service_requires_queue_and_native_boundary")
    if execution.get("kind") == "completion_floor":
        if set(execution) != {"kind"}:
            raise ModelingError("service_regime_completion_floor_execution_requires_only_kind")
        return None
    if execution.get("kind") == "mac_service":
        if (set(execution) != MAC_EXECUTION_FIELDS
                or execution.get("resource") != "cube"
                or not execution.get("native_kernel")):
            raise ModelingError("mac_service_requires_cube_and_native_kernel")
        for key in ("tile_m", "tile_n", "tile_k", "slots"):
            value = execution.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelingError("mac_service_requires_positive_tile_geometry")
        return {**execution}
    if (set(execution) != EXECUTION_FIELDS or execution.get("kind") != "joint_native"
            or not execution.get("resource") or not execution.get("native_kernel")
            or _nonnegative(execution.get("reference_rate", 0)) == 0):
        raise ModelingError("joint_native_service_requires_queue_and_native_boundary")
    return {**execution}


def _axes(context, *, physical_rates=False):
    op = context.op
    axes = {**context.service_identity,
            "hardware": context.service_identity.get("hardware", {}).get("key"),
            **{key: getattr(op, key) for key in ("recipe", "family", "phase", "local_shape",
                                                "semantic_shape", "padded_shape", "launch_shape")},
            "metadata": dict(op.metadata), "work": context.native_work,
            "resource_units": context.resource_units, "core_count": context.core_count}
    if physical_rates:
        # A joint native duration cannot identify responses to individual pipe
        # changes. Explicit rate guards keep it in its declared hardware domain.
        # Resource rates are per second; compute peaks and mode rates already
        # use their respective native work units per second.
        axes["physical_rates"] = {
            "resources": {role: context.rates[name] * 1000
                          for role, name in context.resource_names.items()},
            "compute_peaks": context.compute_peaks,
            "service_modes": context.service_modes,
        }
    return axes


def _rate(context, declaration):
    role = declaration["resource"]
    if role == "cube":
        return context.compute_peaks[declaration["dtype"]]
    service_rate = context.service_rate(role)
    if service_rate is not None:
        return service_rate
    if role in context.resource_names:
        return context.rates[context.resource_names[role]] * 1000
    if role == "link":
        # Shared routes may traverse several physical links. A declared link
        # response uses the bottleneck encountered by this prepared invocation.
        names = set(context.work) - set(context.resource_names.values())
        if names:
            return min(context.rates[name] for name in names) * 1000
    raise ModelingError("service_regime_scaling_resource_missing:" + role)


class RegimePolicy:
    def __init__(self, declarations=()):
        declarations = tuple(declarations)
        self.rules = []
        self.responses = {}
        self.executions = {}
        self.physical_rates = False
        ids = set()
        for declaration in declarations:
            raw = deepcopy(declaration)
            if not isinstance(raw, Mapping) or set(raw) - RULE_FIELDS:
                raise ModelingError("service_regime_unknown_fields")
            if not raw.get("id") or raw["id"] in ids:
                raise ModelingError("service_regime_requires_unique_id")
            ids.add(raw["id"])
            for field in ("match", "coordinates", "domain", "provenance"):
                if field in raw and not isinstance(raw[field], Mapping):
                    raise ModelingError("service_regime_requires_mapping:" + field)
            if "response" not in raw:
                raise ModelingError("service_regime_response_requires_mapping")
            execution = _execution(raw)
            for path in raw.get("match", {}):
                self.physical_rates |= _path(path)[0] == "physical_rates"
            for coord in raw.get("coordinates", {}).values():
                if not isinstance(coord, Mapping):
                    raise ModelingError("service_regime_coordinate_unknown_reducer")
                self.physical_rates |= _path(coord["path"])[0] == "physical_rates"
                if (set(coord) - {"path", "reduce", "scale", "default"}
                        or coord.get("reduce", "sum") not in REDUCERS):
                    raise ModelingError("service_regime_coordinate_unknown_reducer")
                _nonnegative(coord.get("scale", 1))
                if "default" in coord:
                    _nonnegative(coord["default"])
            for name, bounds in raw.get("domain", {}).items():
                if (name not in raw.get("coordinates", {})
                        or set(bounds) - {"min", "max", "min_inclusive", "max_inclusive"}):
                    raise ModelingError("service_regime_domain_requires_declared_coordinate")
                for flag in ("min_inclusive", "max_inclusive"):
                    if flag in bounds and not isinstance(bounds[flag], bool):
                        raise ModelingError("service_regime_domain_inclusive_flag_requires_boolean")
                low = _nonnegative(bounds.get("min", 0))
                high = _nonnegative(bounds["max"]) if "max" in bounds else None
                if high is not None and high < low:
                    raise ModelingError("service_regime_domain_reversed")
            response = ResponseFunction.from_mapping(raw["response"], raw.get("coordinates", {}))
            raw.setdefault("provenance", {})
            key = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
            self.rules.append((raw, key))
            self.responses[key] = response
            self.executions[key] = execution
        # Static compilation is the single interpretation of coverage. It is
        # shared with profiler importers instead of being re-implemented there.
        compiled = validate_service_regime_domains(declarations)
        coverage_by_id = {
            rule_id: bounds
            for report in compiled
            for rule_id, bounds in report.get("coverage", {}).items()
        }
        self.compiled_domains = {}
        for raw, key in self.rules:
            declared = dict(raw.get("domain", {}))
            if raw["id"] in coverage_by_id:
                declared.update(coverage_by_id[raw["id"]])
            self.compiled_domains[key] = declared
        self.groups = self._groups()

    def _groups(self):
        groups = {}
        for raw, key in self.rules:
            continuity = raw.get("continuity")
            if continuity is None:
                continue
            group = groups.setdefault(continuity["group"], {
                "group": continuity["group"], "match": raw.get("match", {}),
                "coordinates": raw.get("coordinates", {}),
                "support": continuity["support"],
                "outside": continuity.get("outside", "analytic"), "members": []})
            group["members"].append((raw, key))
        return tuple(groups.values())

    def select(self, context, record):
        """Native multi-phase groups keep their source/transport program."""
        selected = self.resolve(context, record)
        if selected is not None and selected.mode != "completion_floor":
            raise ModelingError(
                "joint_native_or_mac_service_requires_single_local_kernel")
        return selected.duration_ms if selected is not None else 0.0

    def resolve(self, context, record):
        if context.service_boundary != "device_kernel" or context.op.family in {"barrier", "token_commit"}:
            return None
        axes, matches = _axes(context, physical_rates=self.physical_rates), []
        for raw, key in self.rules:
            selectors = raw.get("match", {})
            if any(axes.get(k) != selectors[k] for k in ("recipe", "kernel", "family") if k in selectors):
                continue
            mismatches = [path for path, value in selectors.items()
                          if not axis_equal(lookup(axes, path), value)]
            if mismatches:
                # Report a related operator's physical-domain miss, without
                # creating per-invocation logs for unrelated kernel families.
                kinds = selectors.keys() & {"recipe", "kernel", "family"}
                if kinds and not kinds.intersection(mismatches):
                    record(raw, key, "physical_identity_mismatch", mismatched_axes=mismatches)
                continue
            values = coordinates(axes, raw.get("coordinates", {}))
            if values is None:
                # A coverage contract declares its coordinates as required: a
                # qualified member must never fall through to analytics. An
                # isolated legacy box remains auditable applicability evidence.
                if raw.get("continuity") is not None:
                    raise ModelingError("service_regime_requires_declared_coordinate:" + raw["id"])
                record(raw, key, "missing_coordinate")
                continue
            if not _within(values, self.compiled_domains[key]):
                # A non-support guard exclusion is reported by the exterior
                # decision; a coverage-dimension exclusion remains a domain
                # audit for legacy and malformed declarations.
                if raw.get("continuity") is None or not _guard_excluded(raw, values):
                    record(raw, key, "analytic_outside_domain", coordinate_values=values)
                continue
            result = self.responses[key].evaluate(values, rate_for=lambda d: _rate(context, d),
                resource_for=lambda d: context.resource_names.get(
                    context.physical_role(d["resource"]), d["resource"]))
            if result.duration_us is None:
                if raw.get("continuity") is not None:
                    raise ModelingError(
                        "service_regime_response_outside_declared_coverage:" + raw["id"])
                record(raw, key, result.basis, coordinate_values=values)
                continue
            matches.append((raw, key, result, values))
        if len(matches) > 1:
            owners = [match for match in matches
                      if _seam_owner(match[0], match[1], match[3], matches, self.compiled_domains)]
            if len(owners) != 1:
                raise ModelingError("service_regime_overlapping_domains:"
                                    + ",".join(r[0]["id"] for r in matches))
            matches = owners
        if not matches:
            return self._exterior(context, axes, record)
        raw, key, result, values = matches[0]
        execution = self.executions[key]
        continuity = raw.get("continuity") or {}
        mode = str(execution.get("kind")) if execution is not None else "completion_floor"
        record(raw, key, ("joint_native_service" if mode == "joint_native"
                          else "observable_response"),
               duration_us=result.duration_us, coordinate_values=values,
               response_basis=result.basis, continuity_group=continuity.get("group"),
               execution_mode=mode, issued_work=result.issued_work,
               rate_flops_per_s=result.rate_flops_per_s,
               active_owners=result.active_owners, stage=result.stage)
        return RegimeSelection(
            result.duration_us / 1000, execution, raw["id"],
            continuity.get("group"), result.basis, values,
            issued_work=result.issued_work, rate_flops_per_s=result.rate_flops_per_s,
            active_owners=result.active_owners, stage=result.stage,
            provenance=dict(raw.get("provenance") or {}))

    def _exterior(self, context, axes, record):
        """Apply one declared exterior policy; never clamp or silently select."""
        candidates, guard_excluded = [], []
        for group in self.groups:
            if any(not axis_equal(lookup(axes, path), value)
                   for path, value in group["match"].items()):
                continue
            values = coordinates(axes, group["coordinates"])
            if values is None:
                raise ModelingError("service_regime_group_requires_declared_coordinate:"
                                    + group["group"])
            excluded = any(_guard_excluded(raw, values) for raw, _ in group["members"])
            if all(_guard(values[name], bounds) for name, bounds in group["support"].items()):
                # A non-support guard may intentionally narrow applicability;
                # it is an exterior decision, not a coverage hole. Otherwise
                # the compiled partition promised a response at this point.
                if not excluded:
                    raise ModelingError("service_regime_domain_interior_gap:" + group["group"])
            if excluded:
                guard_excluded.append((group, values))
            else:
                candidates.append((group, values))
        if not candidates:
            for group, values in guard_excluded:
                raw, key = group["members"][0]
                mode = "joint_native" if self.executions[key] is not None else "completion_floor"
                record(raw, key, "analytic_outside_support", coordinate_values=values,
                       continuity_group=group["group"], execution_mode=mode,
                       exterior_policy="analytic")
            return None
        for group, _ in candidates:
            if group["outside"] == "error":
                raise ModelingError("service_regime_coordinate_outside_support:" + group["group"])
        if len(candidates) > 1 and any(group["outside"] != "analytic" for group, _ in candidates):
            raise ModelingError("service_regime_ambiguous_exterior_groups:"
                                + ",".join(group["group"] for group, _ in candidates))
        if all(group["outside"] == "analytic" for group, _ in candidates):
            for group, values in candidates:
                raw, key = group["members"][0]
                mode = "joint_native" if self.executions[key] is not None else "completion_floor"
                record(raw, key, "analytic_outside_support", coordinate_values=values,
                       continuity_group=group["group"], execution_mode=mode,
                       exterior_policy="analytic")
            return None
        group, values = candidates[0]
        raw, key, result = self._nearest(context, group, values)
        execution = self.executions[key]
        mode = "joint_native" if execution is not None else "completion_floor"
        record(raw, key, "linear_outside_support", duration_us=result.duration_us,
               coordinate_values=values, response_basis=result.basis,
               continuity_group=group["group"], execution_mode=mode,
               exterior_policy="linear")
        return RegimeSelection(result.duration_us / 1000, execution, raw["id"],
                               group["group"], result.basis, values, "linear",
                               provenance=dict(raw.get("provenance") or {}))

    def _nearest(self, context, group, values):
        ranked = []
        for raw, key in group["members"]:
            declared = self.compiled_domains[key]
            distance = 0.0
            for name in group["support"]:
                bounds = declared.get(name, {})
                distance += max(bounds.get("min", 0) - values[name], 0)
                distance += max(values[name] - bounds.get("max", math.inf), 0)
            ranked.append((distance, raw["id"], raw, key))
        best_distance = min(item[0] for item in ranked)
        nearest = [item for item in ranked if item[0] == best_distance]
        nearest.sort(key=lambda item: item[1])
        evaluated = []
        for _, _, raw, key in nearest:
            result = self.responses[key].evaluate(
                values, rate_for=lambda d: _rate(context, d),
                resource_for=lambda d: context.resource_names.get(
                    context.physical_role(d["resource"]), d["resource"]))
            if result.duration_us is None:
                raise ModelingError("service_regime_linear_outside_support_unavailable:" + raw["id"])
            evaluated.append((raw, key, result))
        first_us = evaluated[0][2].duration_us
        for raw, _, result in evaluated[1:]:
            scale = max(abs(first_us), abs(result.duration_us))
            if abs(first_us - result.duration_us) > 1e-6 + 1e-9 * scale:
                raise ModelingError("service_regime_ambiguous_exterior_extrapolation:"
                                    + group["group"] + ":" + evaluated[0][0]["id"] + ":" + raw["id"])
        return evaluated[0]


def assess_regime_migration(declarations, *, resource_services=(), links=None,
                            hardware=None):
    """Return T20 readiness from explicit legacy-work -> service bindings.

    Old provenance is diagnostic evidence, not an admission gate for a new
    identified service. A rule is ready only when every declared work term and
    its fixed component has a complete binding, the service's own conditions are
    present and satisfied, and no binding conflicts remain. The evaluation is
    read-only and is never consumed by ``Execution``.
    """
    policy = RegimePolicy(declarations)
    by_id = {raw["id"] for raw, _ in policy.rules}
    services = {}
    for service in resource_services or ():
        record = normalize_service_handoff(service)
        if record["service_id"] in services:
            raise ModelingError("migration_resource_service_id_not_unique:"
                                + record["service_id"])
        services[record["service_id"]] = record
    plans, legacy_links = _split_links(links)
    unknown = sorted((set(plans) | set(legacy_links)) - by_id)
    if unknown:
        raise ModelingError("migration_link_unknown_regime_rule:" + ",".join(unknown))
    for rule_id, ids in legacy_links.items():
        missing = [sid for sid in ids if sid not in services]
        if missing:
            raise ModelingError("migration_link_unknown_resource_service:"
                                + ",".join(missing))
    records = []
    for raw, key in policy.rules:
        rule_id = raw["id"]
        decomposition = policy.responses[key].service_decomposition()
        execution = policy.executions[key]
        mode = str(execution.get("kind")) if execution is not None else "completion_floor"
        old = _provenance_identifiability(raw.get("provenance") or {})
        old_roles = provenance_roles(raw.get("provenance") or {})
        old_tokens = (old_roles["training"] or old_roles["other"]) if old["negative"] else []
        plan = plans.get(rule_id)
        if plan is None:
            candidates = legacy_links.get(rule_id)
            if candidates:
                record = _no_plan_record(raw, decomposition, mode, old)
                record.update({
                    "status": "awaiting_binding_plan",
                    "target_authority": "resource_rate_service",
                    "linked_services": list(candidates),
                    "blockers": ["migration_binding_plan_required",
                                 "t06_normalized_resource_service_not_linked"],
                    "requires": ["T06", "T04"],
                })
            else:
                record = _no_plan_record(raw, decomposition, mode, old)
        else:
            normalized = normalize_migration_plan(plan, decomposition)
            evaluation = evaluate_migration_binding(
                decomposition, raw.get("match") or {}, rule_id, normalized, services,
                execution=execution,
                old_provenance_negative=old["negative"],
                old_training_tokens=old_tokens,
                coordinate_declarations=raw.get("coordinates") or {})
            status = _PUBLIC_STATUS[evaluation["status"]]
            kind = normalized["replacement_kind"]
            target = "diagnostic_observation_replay"
            if status in {"resource_phase_ready", "slice_ready", "partial",
                          "awaiting_binding_plan", "awaiting_conditions",
                          "awaiting_program_binding"}:
                target = _target_authority(kind)
            if status in {"conflict", "link_rejected"}:
                target = "unresolved"
            blockers = list(evaluation["blockers"])
            if (not decomposition.get("terms")
                    and not (evaluation["coverage"].get("external_work") or ())):
                status, target = "observation_replay", "diagnostic_observation_replay"
                blockers.append("actual_stage_work_reference_required")
            if status in {"resource_phase_ready", "slice_ready"}:
                blockers.append("t04_executor_switch_pending")
            requires = (["T04"] if status in {"resource_phase_ready", "slice_ready"}
                        else ["T06", "T04"] if status in {
                            "awaiting_binding_plan", "awaiting_conditions",
                            "awaiting_program_binding", "partial"}
                        else [])
            record = {
                "rule_id": rule_id, "execution_mode": mode,
                "response_class": decomposition["response_class"],
                "production_authority": {
                    "completion_floor": "legacy_completion_floor",
                    "joint_native": "legacy_joint_native_busy_ns",
                    "mac_service": "legacy_mac_service"}.get(mode, "legacy_" + mode),
                "target_authority": target, "status": status,
                "replacement_scope": evaluation["replacement_scope"],
                "decomposition": decomposition,
                "linked_services": sorted({item["service_id"]
                                           for item in normalized["work"]}
                                          | {item["service_id"]
                                             for item in normalized["fixed"]}),
                "blockers": sorted(set(blockers)),
                "provenance_evidence": list(old["evidence"]),
                "coverage": evaluation["coverage"],
                "conditions": evaluation["conditions"],
                "diagnostics": evaluation["diagnostics"],
                "service_evidence": evaluation["service_evidence"],
                "scope": evaluation["scope"], "requires": requires,
            }
        records.append(record)
    summary = {status: sum(record["status"] == status for record in records)
               for status in sorted(MIGRATION_STATUSES)}
    return {
        "schema": MIGRATION_SCHEMA, "role": "diagnostic_migration_assessment",
        "production_prediction": False, "production_switch_performed": False,
        "switch_gate": {"status": "pending_t18_rolling_chain",
                        "requires": ["T04", "T06", "T18", "T19"],
                        "note": "readiness is not a production switch; T19 must "
                                "validate the candidate after T18/T20"},
        "hardware": hardware, "summary": {"rules": len(records), **summary},
        "rules": records,
    }


def resource_phase_bindings(declarations, *, resource_services=(), links=None):
    """Return concrete T04/T17 handoff bindings for ready rules only.

    Each binding carries the rule, service evidence, work-term coverage,
    multiplicity, fixed ownership and the old-evidence diagnostics. It is a
    read-only producer handoff, not a second execution mode.
    """
    readiness = assess_regime_migration(
        declarations, resource_services=resource_services, links=links)
    bindings = [record for record in readiness["rules"]
                if record["status"] in {"resource_phase_ready", "slice_ready"}
                and record.get("coverage") and record["coverage"].get("complete")
                and record.get("service_evidence")]
    return {
        "schema": HANDOFF_SCHEMA, "role": "resource_phase_binding_handoff",
        "production_prediction": False, "production_switch_performed": False,
        "switch_gate": readiness["switch_gate"],
        "readiness": readiness["summary"], "bindings": bindings,
    }

__all__ = ["RegimePolicy", "RegimeSelection", "assess_regime_migration",
           "resource_phase_bindings", "normalize_service_handoff",
           "MIGRATION_SCHEMA", "HANDOFF_SCHEMA"]
