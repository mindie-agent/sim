"""Observable small-workload response regimes on the common Engine.

[Role] Bind declared per-call/group/byte responses to physical kernel domains.
[Invariants] A response is inclusive, not an added whole-body residual. The
  default completion floor retains native work. Explicit joint-native service
  replaces one local kernel schedule; its analytical work remains a reference.
  One native group receives one response, never one per internal phase.
  Applicability, numerical response, coverage and execution stay separate:
  omitted selectors mean broader applicability, a non-response coordinate may
  be a bounded guard, and a continuity group must partition its declared
  support exactly once (compiled by service_regime_domains). Calibration
  observations and validation roles remain provenance, not simulation admission.
[Timeouts & Locks] Finite rule list and bounded aggregate records; no I/O/locks.
[Side-Effect Policy] Read physical axes and return an explicit service selection.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math

from mindie_sim.schema.common import ModelingError
from .response_functions import ResponseFunction, nonnegative as _nonnegative, scaled_coordinate
from .service_regime_domains import validate_service_regime_domains


ROOTS = {"hardware", "runtime", "revision", "format", "kernel", "implementation",
         "dtype", "layout", "replay_scope", "recipe", "family", "phase",
         "local_shape", "semantic_shape", "padded_shape", "launch_shape",
         "metadata", "work", "resource_units", "core_count", "physical_rates", "transport"}
REDUCERS = {"sum": sum, "max": max, "min": min, "len": len,
            "product": math.prod, "positive_count": lambda x: sum(v > 0 for v in x)}
MISSING = object()
RULE_FIELDS = {"id", "match", "coordinates", "domain", "response", "provenance",
               "execution", "continuity"}
EXECUTION_FIELDS = {"kind", "resource", "reference_rate", "native_kernel"}


@dataclass(frozen=True)
class RegimeSelection:
    duration_ms: float
    execution: Mapping | None = None
    rule_id: str | None = None
    continuity_group: str | None = None
    response_basis: str | None = None
    coordinate_values: Mapping | None = None
    exterior_policy: str | None = None

    @property
    def mode(self):
        """Normalized execution semantics, never an arbitrary mapping probe."""
        return "joint_native" if self.execution is not None else "completion_floor"

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
            return None
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
    if role in context.service_modes:
        return context.service_modes[role][1]
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
                if set(coord) - {"path", "reduce", "scale"} or coord.get("reduce", "sum") not in REDUCERS:
                    raise ModelingError("service_regime_coordinate_unknown_reducer")
                _nonnegative(coord.get("scale", 1))
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
            raise ModelingError("joint_native_service_requires_single_local_kernel")
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
        mode = "joint_native" if execution is not None else "completion_floor"
        record(raw, key, "joint_native_service" if execution else "observable_response",
               duration_us=result.duration_us, coordinate_values=values,
               response_basis=result.basis, continuity_group=continuity.get("group"),
               execution_mode=mode)
        return RegimeSelection(result.duration_us / 1000, execution, raw["id"],
                               continuity.get("group"), result.basis, values)

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
                               group["group"], result.basis, values, "linear")

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
