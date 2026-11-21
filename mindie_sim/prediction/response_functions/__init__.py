"""Pure physical-coordinate response functions shared by execution and scoring.

[Role] Evaluate physical supply responses and declared interpolation surfaces.
  Also expose the declared coefficient structure needed to decide whether a
  legacy response can be replaced by a resource/phase service.
[Invariants] Knots are supplied parameters, never learned from query timings.
  Every coordinate, scaled coordinate, coefficient, knot, grid value and rate
  is finite and nonnegative; booleans are never numbers and rates are strictly
  positive before division. A grid requires every corner. Extrapolation never
  clamps an invalid duration. Response dependencies, knot support and cell
  breakpoints are exposed for static coverage compilation; continuity never
  implies global monotonicity. Coefficient decomposition is diagnostic
  evidence; it never evaluates a query or changes the response formula.
[Timeouts & Locks] Bounded parameter tables and binary search; no waits or locks.
[Side-Effect Policy] No I/O, fitting, global mutation or device execution.
"""
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import math

from mindie_sim.schema.common import ModelingError

from .evaluation import (
    ResponseFunction,
    ResponseValue,
    nonnegative,
    positive,
    scaled_coordinate,
    _duration,
    _ordered_knots,
    _outside,
    _pairs,
    _value,
)
from .migration import (
    MIGRATION_FIXED_FIELDS,
    MIGRATION_PLAN_FIELDS,
    MIGRATION_REPLACEMENT_KINDS,
    MIGRATION_SCOPES,
    MIGRATION_WORK_FIELDS,
    normalize_migration_plan,
    provenance_roles,
    same_declared_value,
    _OTHER_SOURCE_FIELDS,
    _TRAINING_FIELDS,
    _VALIDATION_FIELDS,
    _WORK_SCOPES,
    _fragment_evidence,
    _plan_text,
    _program_view,
    _role_tokens,
)


__all__ = [
    "MIGRATION_FIXED_FIELDS",
    "MIGRATION_PLAN_FIELDS",
    "MIGRATION_REPLACEMENT_KINDS",
    "MIGRATION_SCOPES",
    "MIGRATION_WORK_FIELDS",
    "Mapping",
    "ModelingError",
    "ResponseFunction",
    "ResponseValue",
    "bisect_left",
    "dataclass",
    "math",
    "nonnegative",
    "normalize_migration_plan",
    "positive",
    "product",
    "provenance_roles",
    "same_declared_value",
    "scaled_coordinate",
]
