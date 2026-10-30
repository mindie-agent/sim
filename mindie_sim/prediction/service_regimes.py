"""Observable small-workload response regimes on the common Engine.

[Role] Bind declared per-call/group/byte responses to physical kernel domains.
[Invariants] A response is inclusive, not an added whole-body residual. The
  default completion floor retains native work. Explicit joint-native service
  replaces one local kernel schedule; its analytical work remains a reference.
  One native group receives one response, never one per internal phase. Calibration observations
  and validation roles remain explicit provenance, not simulation admission.
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
from .response_functions import ResponseFunction, nonnegative as _nonnegative


ROOTS = {"hardware", "runtime", "revision", "format", "kernel", "implementation",
         "dtype", "layout", "replay_scope", "recipe", "family", "phase",
         "local_shape", "semantic_shape", "padded_shape", "launch_shape",
         "metadata", "work", "resource_units", "core_count", "physical_rates", "transport"}
REDUCERS = {"sum": sum, "max": max, "min": min, "len": len,
            "product": math.prod, "positive_count": lambda x: sum(v > 0 for v in x)}
MISSING = object()


@dataclass(frozen=True)
class RegimeSelection:
    duration_ms: float
    execution: Mapping | None = None

    def bind_joint(self, context):
        """Observed joint service occupies an explicitly declared device queue.

        No byte/cycle occupancy is identified by a complete kernel duration.
        The analytical schedule remains a reference, not simultaneous demand.
        """
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
        result[name] = _nonnegative(value) * raw.get("scale", 1)
    return result


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
        self.rules = []
        self.responses = {}
        self.physical_rates = False
        ids = set()
        for declaration in declarations:
            raw = deepcopy(declaration)
            if set(raw) - {"id", "match", "coordinates", "domain", "response", "provenance", "execution"}:
                raise ModelingError("service_regime_unknown_fields")
            if not raw.get("id") or raw["id"] in ids:
                raise ModelingError("service_regime_requires_unique_id")
            ids.add(raw["id"])
            execution = raw.get("execution")
            if execution is not None:
                if (set(execution) != {"kind", "resource", "reference_rate", "native_kernel"}
                        or execution["kind"] != "joint_native"
                        or not execution["resource"] or not execution["native_kernel"]
                        or _nonnegative(execution["reference_rate"]) == 0):
                    raise ModelingError("joint_native_service_requires_queue_and_native_boundary")
            for path in raw.get("match", {}):
                self.physical_rates |= _path(path)[0] == "physical_rates"
            for coord in raw.get("coordinates", {}).values():
                self.physical_rates |= _path(coord["path"])[0] == "physical_rates"
                if set(coord) - {"path", "reduce", "scale"} or coord.get("reduce", "sum") not in REDUCERS:
                    raise ModelingError("service_regime_coordinate_unknown_reducer")
                _nonnegative(coord.get("scale", 1))
            for name, bounds in raw.get("domain", {}).items():
                if name not in raw.get("coordinates", {}) or set(bounds) - {"min", "max"}:
                    raise ModelingError("service_regime_domain_requires_declared_coordinate")
                low = _nonnegative(bounds.get("min", 0))
                high = _nonnegative(bounds["max"]) if "max" in bounds else None
                if high is not None and high < low:
                    raise ModelingError("service_regime_domain_reversed")
            response = ResponseFunction.from_mapping(raw["response"], raw.get("coordinates", {}))
            if response.kind in {"piecewise_linear", "multilinear", "roofline"}:
                selectors = raw.get("match", {})
                identity = {"hardware", "dtype", "layout", "replay_scope", "implementation"}
                if response.kind != "roofline":
                    identity |= {"runtime", "revision", "format"}
                if (not identity <= selectors.keys()
                        or not selectors.keys() & {"recipe", "kernel"}):
                    raise ModelingError("service_regime_curve_requires_fixed_execution_identity")
            if response.kind in {"piecewise_linear", "multilinear"}:
                # An empirical table identifies only its declared axes. Other
                # numerical inputs remain fixed conditions, not wildcards.
                for name in raw.get("coordinates", {}).keys() - response.varying_coordinates:
                    bounds = raw.get("domain", {}).get(name, {})
                    if "min" not in bounds or "max" not in bounds or bounds["min"] != bounds["max"]:
                        raise ModelingError("service_regime_curve_requires_fixed_other_coordinates:" + name)
            raw.setdefault("provenance", {})
            key = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
            self.rules.append((raw, key))
            self.responses[key] = response

    def select(self, context, record):
        """Native multi-phase groups keep their source/transport program."""
        selected = self.resolve(context, record)
        if selected is not None and selected.execution is not None:
            raise ModelingError("joint_native_service_requires_single_local_kernel")
        return selected.duration_ms if selected is not None else 0.0

    def resolve(self, context, record):
        if context.service_boundary != "device_kernel" or context.op.family in {"barrier", "token_commit"}:
            return None
        axes, matches = _axes(context, physical_rates=self.physical_rates), []
        for raw, key in self.rules:
            selectors = raw.get("match", {})
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
                record(raw, key, "missing_coordinate")
                continue
            if any(not bounds.get("min", 0) <= values[name] <= bounds.get("max", math.inf)
                   for name, bounds in raw.get("domain", {}).items()):
                record(raw, key, "analytic_outside_domain", coordinate_values=values)
                continue
            result = self.responses[key].evaluate(values, rate_for=lambda d: _rate(context, d),
                resource_for=lambda d: context.resource_names.get(
                    context.physical_role(d["resource"]), d["resource"]))
            if result.duration_us is None:
                record(raw, key, result.basis, coordinate_values=values)
                continue
            matches.append((raw, key, result, values))
        if len(matches) > 1:
            raise ModelingError("service_regime_overlapping_domains:" + ",".join(r[0]["id"] for r in matches))
        if not matches:
            return None
        raw, key, result, values = matches[0]
        record(raw, key, "joint_native_service" if raw.get("execution") else "observable_response",
               duration_us=result.duration_us, coordinate_values=values, response_basis=result.basis)
        return RegimeSelection(result.duration_us / 1000, raw.get("execution"))
