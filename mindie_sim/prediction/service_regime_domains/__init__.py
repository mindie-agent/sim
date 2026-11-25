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
from ..response_functions import (
    ResponseFunction,
    nonnegative,
    positive,
)

from .geometry import (
    ATOL_US,
    EXECUTION_MODES,
    Interval,
    MAX_FACE_VERTICES,
    MAX_PROBES,
    OUTSIDE_POLICIES,
    RTOL,
    RuleDomain,
    compile_service_regime_domains,
    rules_overlap,
    validate_service_regime_domains,
    _BOUND_FIELDS,
    _CONTINUITY_FIELDS,
    _agree,
    _canonical,
    _check_seams,
    _contains,
    _continuity,
    _coverage,
    _dimension_probes,
    _execution,
    _face_samples,
    _flag,
    _interval,
    _limit,
    _normalize,
    _number,
    _overlaps_support,
    _owners,
    _owns,
    _probe_coverage,
    _rate_dependent,
    _response,
    _shared_face,
)


__all__ = [
    "ATOL_US",
    "EXECUTION_MODES",
    "Interval",
    "MAX_FACE_VERTICES",
    "MAX_PROBES",
    "Mapping",
    "ModelingError",
    "OUTSIDE_POLICIES",
    "RTOL",
    "ResponseFunction",
    "RuleDomain",
    "annotations",
    "compile_service_regime_domains",
    "dataclass",
    "json",
    "math",
    "nonnegative",
    "positive",
    "product",
    "rules_overlap",
    "validate_service_regime_domains",
]
