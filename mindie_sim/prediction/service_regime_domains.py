"""Static service-regime domain compilation and coverage validation.

[Role] Normalize declared response domains into physical intervals and prove
  that every continuity group covers its announced support exactly once.
[Invariants] Coverage is geometric: interior gaps, fractional gaps such as
  (128, 129), multidimensional holes, ambiguous overlaps, uncovered faces and
  knot support that cannot reach the declared domain are rejected. Tolerance
  (atol=1e-6 us, rtol=1e-9) compares response values only, never geometry.
  Shared-face limits must agree; rate-dependent or roofline seams require a
  provably equivalent expression. Group members agree on categorical
  selectors, coordinate definitions, execution mode and native boundary.
  Validation work is bounded and fails explicitly when the budget is exceeded.
[Timeouts & Locks] Bounded interval subdivisions; no waits, I/O or locks.
[Side-Effect Policy] Pure validation: reads declarations and returns reports.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import json
import math

from mindie_sim.schema.common import ModelingError
from .response_functions import ResponseFunction, nonnegative

ATOL_US = 1e-6
RTOL = 1e-9
MAX_PROBES = 100_000
MAX_FACE_VERTICES = 4_096
EXECUTION_MODES = frozenset({"completion_floor", "joint_native"})
OUTSIDE_POLICIES = frozenset({"analytic", "linear", "error"})
_BOUND_FIELDS = frozenset({"min", "max", "min_inclusive", "max_inclusive"})
_CONTINUITY_FIELDS = frozenset({"group", "support", "outside"})


@dataclass(frozen=True)
class Interval:
    """One closed (or explicitly half-open) physical interval."""

    name: str
    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @property
    def empty(self):
        if self.lower > self.upper:
            return True
        return self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive)

    def intersect(self, other):
        lower, lower_inclusive = self.lower, self.lower_inclusive
        if other.lower > lower:
            lower, lower_inclusive = other.lower, other.lower_inclusive
        elif other.lower == lower:
            lower_inclusive = lower_inclusive and other.lower_inclusive
        upper, upper_inclusive = self.upper, self.upper_inclusive
        if other.upper < upper:
            upper, upper_inclusive = other.upper, other.upper_inclusive
        elif other.upper == upper:
            upper_inclusive = upper_inclusive and other.upper_inclusive
        return Interval(self.name, lower, upper, lower_inclusive, upper_inclusive)

    def contains(self, value):
        if value < self.lower or value > self.upper:
            return False
        if value == self.lower and not self.lower_inclusive:
            return False
        if value == self.upper and not self.upper_inclusive:
            return False
        return True

    def to_dict(self):
        return {
            "min": self.lower if math.isfinite(self.lower) else None,
            "max": self.upper if math.isfinite(self.upper) else None,
            "min_inclusive": self.lower_inclusive,
            "max_inclusive": self.upper_inclusive,
        }


@dataclass(frozen=True)
class RuleDomain:
    """Normalized applicability, response and execution of one rule."""

    rule_id: str
    index: int
    match: Mapping
    coordinates: Mapping
    execution_mode: str
    execution: Mapping | None
    group: str
    support: Mapping[str, Interval]
    function: ResponseFunction
    response: Mapping
    outside: str
    coverage: Mapping[str, Interval]
    guards: Mapping[str, Interval]

    def to_dict(self):
        return {"id": self.rule_id, "index": self.index,
                "execution_mode": self.execution_mode,
                "response_basis": self.response.get("kind", "affine"),
                "outside": self.outside,
                "coverage": {name: interval.to_dict()
                             for name, interval in self.coverage.items()}}


def _canonical(value):
    return json.dumps(value, sort_keys=True, default=repr, allow_nan=False,
                      separators=(",", ":"))


def _number(value, code):
    try:
        return nonnegative(value)
    except ModelingError as exc:
        raise ModelingError(code) from exc


def _flag(bounds, field):
    value = bounds.get(field, True)
    if not isinstance(value, bool):
        raise ModelingError("service_regime_domain_inclusive_flag_requires_boolean:" + field)
    return value


def _interval(name, bounds, *, require_max=False):
    if not isinstance(bounds, Mapping):
        raise ModelingError("service_regime_domain_bounds_require_mapping:" + name)
    if set(bounds) - _BOUND_FIELDS:
        raise ModelingError("service_regime_domain_unknown_fields:" + name)
    if require_max and ({"min", "max"} - set(bounds)):
        raise ModelingError("service_regime_continuity_support_requires_bounds:" + name)
    lower = _number(bounds.get("min", 0), "service_regime_domain_requires_finite_bounds:" + name)
    if "max" in bounds:
        upper = _number(bounds["max"], "service_regime_domain_requires_finite_bounds:" + name)
    else:
        upper = math.inf
    lower_inclusive = _flag(bounds, "min_inclusive")
    upper_inclusive = _flag(bounds, "max_inclusive")
    if upper < lower:
        raise ModelingError("service_regime_domain_reversed:" + name)
    return Interval(name, lower, upper, lower_inclusive, upper_inclusive)


def _execution(rule):
    execution = rule.get("execution")
    if execution is None:
        return "completion_floor", None
    if not isinstance(execution, Mapping):
        raise ModelingError("service_regime_execution_requires_mapping")
    kind = execution.get("kind")
    if kind == "completion_floor":
        if set(execution) != {"kind"}:
            raise ModelingError("service_regime_completion_floor_execution_requires_only_kind")
        return "completion_floor", None
    if kind != "joint_native":
        raise ModelingError("service_regime_execution_kind_unknown")
    if not execution.get("resource") or not execution.get("native_kernel"):
        raise ModelingError("joint_native_service_requires_queue_and_native_boundary")
    if _number(execution.get("reference_rate", 0),
               "service_regime_reference_rate_must_be_positive") == 0:
        raise ModelingError("service_regime_reference_rate_must_be_positive")
    return "joint_native", dict(execution)


def _continuity(rule):
    raw = rule.get("continuity")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - _CONTINUITY_FIELDS:
        raise ModelingError("service_regime_continuity_unknown_fields")
    group = raw.get("group")
    if not isinstance(group, str) or not group.strip():
        raise ModelingError("service_regime_continuity_requires_group")
    support = raw.get("support")
    if not isinstance(support, Mapping) or not support:
        raise ModelingError("service_regime_continuity_requires_support")
    outside = raw.get("outside", "analytic")
    if outside not in OUTSIDE_POLICIES:
        raise ModelingError("service_regime_continuity_outside_requires_analytic_linear_or_error")
    return {"group": group,
            "support": {name: _interval(name, bounds, require_max=True)
                        for name, bounds in support.items()},
            "outside": outside}


def _response(rule, coordinates):
    response = rule.get("response")
    if not isinstance(response, Mapping):
        raise ModelingError("service_regime_response_requires_mapping")
    function = ResponseFunction.from_mapping(response, coordinates)
    outside = response.get("outside", "analytic") if function.kind in {
        "piecewise_linear", "multilinear"} else "analytic"
    if function.kind in {"piecewise_linear", "multilinear"} and outside not in OUTSIDE_POLICIES:
        raise ModelingError("service_regime_curve_outside_requires_analytic_linear_or_error")
    return response, function, outside


def _coverage(rule, domain, function, support, outside):
    declared = rule.get("domain") or {}
    if not isinstance(declared, Mapping):
        raise ModelingError("service_regime_domain_requires_mapping")
    if set(declared) - set(domain):
        raise ModelingError("service_regime_domain_requires_declared_coordinate")
    if not set(function.varying_coordinates) <= set(support):
        missing = sorted(set(function.varying_coordinates) - set(support))
        raise ModelingError("service_regime_domain_support_requires_response_coordinate:"
                            + ",".join(missing))
    coverage = {}
    for name, base in support.items():
        bounds = declared.get(name)
        interval = _interval(name, bounds) if bounds is not None else base
        interval = interval.intersect(base)
        limits = function.knot_support.get(name)
        if limits is not None and outside != "linear":
            interval = interval.intersect(Interval(name, limits[0], limits[1]))
        if interval.empty:
            raise ModelingError("service_regime_domain_rule_outside_support:"
                                + str(rule.get("id")) + ":" + name)
        coverage[name] = interval
    # A non-support coordinate is an applicability guard, not a coverage axis:
    # a bounded guard asserts applicability throughout its range, while the
    # support contract covers only the response coordinates.
    guards = {}
    for name in set(domain) - set(support):
        bounds = declared.get(name)
        if bounds is not None:
            guards[name] = _interval(name, bounds)
    return coverage, guards


def _normalize(declarations, index):
    rule = declarations[index]
    if not isinstance(rule, Mapping):
        raise ModelingError("service_regime_requires_mapping")
    if not rule.get("id"):
        raise ModelingError("service_regime_requires_unique_id")
    coordinates = rule.get("coordinates") or {}
    if not isinstance(coordinates, Mapping):
        raise ModelingError("service_regime_coordinates_require_mapping")
    response, function, outside = _response(rule, coordinates)
    execution_mode, execution = _execution(rule)
    continuity = _continuity(rule)
    if continuity is None:
        return None
    if continuity["support"].keys() - coordinates.keys():
        raise ModelingError("service_regime_continuity_support_requires_declared_coordinate")
    if outside != continuity["outside"]:
        raise ModelingError("service_regime_continuity_outside_must_match_response")
    if function.kind in {"affine", "roofline"} and continuity["outside"] != "analytic":
        raise ModelingError("service_regime_continuity_outside_must_match_response")
    coverage, guards = _coverage(rule, coordinates, function, continuity["support"], continuity["outside"])
    return RuleDomain(
        rule_id=str(rule["id"]), index=index, match=rule.get("match") or {},
        coordinates=coordinates, execution_mode=execution_mode, execution=execution,
        group=continuity["group"], support=continuity["support"],
        function=function, response=response, outside=continuity["outside"],
        coverage=coverage, guards=guards)


def _agree(group, domains):
    first = domains[0]
    for domain in domains[1:]:
        for field, value in (("match", domain.match), ("coordinates", domain.coordinates)):
            expected = first.match if field == "match" else first.coordinates
            if _canonical(value) != _canonical(expected):
                raise ModelingError("service_regime_domain_members_differ:"
                                    + group + ":" + field)
        if domain.execution_mode != first.execution_mode:
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":execution_mode")
        if domain.execution_mode == "joint_native" and (
                _canonical(domain.execution) != _canonical(first.execution)):
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":native_boundary")
        if domain.outside != first.outside:
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":outside")
        if dict(domain.support) != dict(first.support):
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":support")
        if dict(domain.guards) != dict(first.guards):
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":guards")


def _dimension_probes(intervals, support):
    lower, upper = support.lower, support.upper
    boundaries = {lower, upper}
    for interval in intervals:
        boundaries.add(min(max(interval.lower, lower), upper))
        boundaries.add(min(max(interval.upper, lower), upper))
    ordered = sorted(boundaries)
    probes = set()
    for position, value in enumerate(ordered):
        previous = ordered[position - 1] if position else None
        following = ordered[position + 1] if position + 1 < len(ordered) else None
        inside_support = support.lower <= value <= support.upper
        if inside_support and support.contains(value):
            probes.add(value)
        left = math.nextafter(value, -math.inf)
        if left > (previous if previous is not None else -math.inf) and left >= lower:
            if support.contains(left) or (left == lower and support.lower_inclusive):
                probes.add(left)
        right = math.nextafter(value, math.inf)
        if right < (following if following is not None else math.inf) and right <= upper:
            if support.contains(right) or (right == upper and support.upper_inclusive):
                probes.add(right)
    return tuple(sorted(probes))


def _contains(domain, point):
    return all(interval.contains(point[name])
               for name, interval in domain.coverage.items())


def _owns(domain, point, members):
    if not _contains(domain, point):
        return False
    for name, interval in domain.coverage.items():
        value = point[name]
        if value != interval.upper or interval.lower == interval.upper:
            continue
        for other in members:
            if other is domain:
                continue
            other_interval = other.coverage[name]
            if (other_interval.lower == value and other_interval.lower_inclusive
                    and _contains(other, point)):
                return False
    return True


def _owners(point, members):
    return [domain for domain in members if _owns(domain, point, members)]


def _probe_coverage(group, domains, support):
    dimensions = tuple(sorted(support))
    per_dimension = [_dimension_probes([domain.coverage[name] for domain in domains],
                                       support[name]) for name in dimensions]
    total = math.prod(len(probes) for probes in per_dimension)
    if total > MAX_PROBES:
        raise ModelingError("service_regime_domain_validation_budget_exceeded:"
                            + group + ":" + str(total))
    for combination in product(*per_dimension):
        point = dict(zip(dimensions, combination))
        owners = _owners(point, domains)
        if not owners:
            raise ModelingError("service_regime_domain_interior_gap:"
                                + group + ":" + _canonical(point))
        if len(owners) > 1:
            raise ModelingError("service_regime_domain_interior_overlap:"
                                + group + ":" + ",".join(d.rule_id for d in owners))
    return total


def _shared_face(first, second, support):
    ranges = {}
    touching = False
    for name in support:
        left, right = first.coverage[name], second.coverage[name]
        lower, upper = max(left.lower, right.lower), min(left.upper, right.upper)
        if lower > upper:
            return None
        if lower == upper and (left.lower < left.upper or right.lower < right.upper):
            touching = True
        ranges[name] = (lower, upper)
    return ranges if touching else None


def _face_samples(domains, name, lower, upper):
    samples = {lower, upper}
    for domain in domains:
        for value in domain.function.breakpoints.get(name, ()):
            if lower <= value <= upper:
                samples.add(value)
    ordered = sorted(samples)
    samples.update((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    return sorted(samples)


def _rate_dependent(response):
    if response.get("kind") == "roofline":
        return True
    return any(term.get("scale_with") for term in response.get("terms", ()))


def _limit(domain, values):
    try:
        result = domain.function.evaluate(values)
    except ModelingError as exc:
        raise ModelingError("service_regime_domain_seam_response_invalid:"
                            + domain.rule_id) from exc
    if result.duration_us is None:
        raise ModelingError("service_regime_domain_seam_uncovered:" + domain.rule_id)
    return result.duration_us


def _check_seams(group, domains, support):
    checked = 0
    for position, first in enumerate(domains):
        for second in domains[position + 1:]:
            ranges = _shared_face(first, second, support)
            if ranges is None:
                continue
            if _canonical(first.response) == _canonical(second.response):
                continue
            if _rate_dependent(first.response) or _rate_dependent(second.response):
                raise ModelingError("service_regime_domain_seam_requires_provable_equivalence:"
                                    + group + ":" + first.rule_id + ":" + second.rule_id)
            axes = {name: _face_samples((first, second), name, *ranges[name])
                    for name in ranges}
            vertices = math.prod(len(samples) for samples in axes.values())
            if vertices > MAX_FACE_VERTICES:
                raise ModelingError("service_regime_domain_validation_budget_exceeded:"
                                    + group + ":seam")
            names = tuple(axes)
            for combination in product(*(axes[name] for name in names)):
                values = dict(zip(names, combination))
                left, right = _limit(first, values), _limit(second, values)
                scale = max(abs(left), abs(right))
                if abs(left - right) > ATOL_US + RTOL * scale:
                    raise ModelingError("service_regime_domain_seam_limit_mismatch:"
                                        + group + ":" + first.rule_id + ":" + second.rule_id)
                checked += 1
    return checked


def _overlaps_support(rule, member_raw, domain):
    if not rules_overlap(rule, member_raw):
        return False
    declared = rule.get("domain") or {}
    for name, interval in domain.support.items():
        bounds = declared.get(name) if isinstance(declared, Mapping) else None
        if bounds is None:
            continue
        if _interval(name, bounds).intersect(interval).empty:
            return False
    return True


def compile_service_regime_domains(declarations):
    """Compile every declared continuity group and return bounded reports.

    Raises ModelingError when a group is inconsistent, when its support is not
    partitioned exactly once by member domains and knot support, or when a
    shared-face limit is not provably continuous.
    """
    if isinstance(declarations, Mapping) or not isinstance(declarations, (list, tuple)):
        raise ModelingError("service_regime_declarations_require_sequence")
    domains, legacy, identifiers = {}, [], set()
    for index, rule in enumerate(declarations):
        if isinstance(rule, Mapping):
            identifier = rule.get("id")
            if identifier in identifiers:
                raise ModelingError("service_regime_requires_unique_id")
            identifiers.add(identifier)
        domain = _normalize(declarations, index)
        if domain is None:
            legacy.append(rule)
        else:
            domains.setdefault(domain.group, []).append(domain)
    groups = []
    for group, members in domains.items():
        _agree(group, members)
        support = members[0].support
        probes = _probe_coverage(group, members, support)
        seams = _check_seams(group, members, support)
        groups.append({
            "group": group,
            "execution_mode": members[0].execution_mode,
            "response_basis": members[0].response.get("kind", "affine"),
            "outside": members[0].outside,
            "support": {name: interval.to_dict() for name, interval in support.items()},
            "guards": {name: interval.to_dict() for name, interval in members[0].guards.items()},
            "members": [domain.rule_id for domain in members],
            "coverage": {domain.rule_id: {name: interval.to_dict()
                                          for name, interval in domain.coverage.items()}
                         for domain in members},
            "probes": probes, "seam_vertices": seams,
        })
    for rule in legacy:
        if not isinstance(rule, Mapping):
            continue
        match = _canonical(rule.get("match") or {})
        for group, members in domains.items():
            if match != _canonical(members[0].match):
                continue
            if any(_overlaps_support(rule, declarations[domain.index], domain)
                   for domain in members):
                raise ModelingError("service_regime_domain_requires_coverage_contract:"
                                    + group + ":" + str(rule.get("id")))
    return tuple(groups)


def validate_service_regime_domains(declarations):
    """Static compatibility/coverage checker exported for profiler importers."""
    return compile_service_regime_domains(declarations)


def rules_overlap(first, second):
    """Conservative static overlap of two declarations on shared coordinates."""
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ModelingError("service_regime_requires_mapping")
    if _canonical(first.get("match") or {}) != _canonical(second.get("match") or {}):
        return False
    left = first.get("coordinates") or {}
    right = second.get("coordinates") or {}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ModelingError("service_regime_coordinates_require_mapping")
    for name in set(left) & set(right):
        first_bounds = (first.get("domain") or {}).get(name)
        second_bounds = (second.get("domain") or {}).get(name)
        if first_bounds is None or second_bounds is None:
            continue
        if _interval(name, first_bounds).intersect(_interval(name, second_bounds)).empty:
            return False
    return True
