"""Discrete structural guards and continuous pipe-competition guards.

[Role]
  Partition a bounded dynamic geometry into discrete implementation regimes
  (query/KV tiles, core waves, split-K, graph buckets) and locate the
  continuous resource crossover between independent physical pipes.

[Invariants]
  * A guard cell owns an explicit half-open interval ``(lower, upper]``.  A
    value outside the declared domain raises ``guard_domain_miss`` instead of
    silently selecting the nearest cell.
  * A bounded audit checks that every lattice point in a domain has exactly one
    matching cell; gaps and overlaps are reported separately from misses.
  * Discrete boundaries come from declared tiles/splits/buckets.  Continuous
    crossovers are roots of declared work/supply expressions, never measured
    thresholds.  A tie records every co-dominant pipe with stable declaration
    order; no artificial dispatch jump is introduced.
  * Service modes that share one physical pipe are summed before the envelope
    max; distinct pipes are allowed to overlap.

[Timeouts & Locks]
  Bounded cell enumeration and closed-form roots.  No scans, waits or locks.

[Side-Effect Policy]
  Pure immutable records and reports; no I/O, global mutation or measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from .expressions import Const, Div, Expr, Sub
from .expressions import as_fraction, evaluate


def _nonnegative_int(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError(reason)
    return value


@dataclass(frozen=True)
class GuardCell:
    """One regime cell on one coordinate: ``lower < value <= upper``."""

    coordinate: str
    index: int
    lower: Fraction | None
    upper: Fraction | None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.coordinate:
            raise ModelingError("guard_cell_coordinate_invalid")
        if self.index < 0:
            raise ModelingError("guard_cell_index_invalid")
        lower = None if self.lower is None else as_fraction(self.lower)
        upper = None if self.upper is None else as_fraction(self.upper)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if lower is not None and upper is not None and lower >= upper:
            raise ModelingError(f"guard_cell_inverted:{self.coordinate}:{self.index}")

    def contains(self, value: Any) -> bool:
        exact = as_fraction(value)
        return ((self.lower is None or exact > self.lower)
                and (self.upper is None or exact <= self.upper))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate, "index": self.index,
            "lower": None if self.lower is None else str(self.lower),
            "upper": None if self.upper is None else str(self.upper),
            "label": self.label,
        }


@dataclass(frozen=True)
class PartitionAudit:
    """Result of a bounded uniqueness/gap audit on one coordinate."""

    coordinate: str
    lower: int
    upper: int
    cells: tuple[GuardCell, ...]
    matches: tuple[tuple[int, int], ...]
    misses: tuple[int, ...] = ()
    overlaps: tuple[int, ...] = ()
    checked: int = 0

    @property
    def unique(self) -> bool:
        return not self.misses and not self.overlaps

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate, "lower": self.lower,
            "upper": self.upper, "checked": self.checked,
            "cells": [cell.to_dict() for cell in self.cells],
            "matches": dict(self.matches), "misses": list(self.misses),
            "overlaps": list(self.overlaps), "unique": self.unique,
        }


@dataclass(frozen=True)
class StepGuard:
    """Uniform ceil step, e.g. ``ceil(qg/M_t)`` or ``ceil(s/N_t)``."""

    name: str
    coordinate: str
    tile: int
    lower_bound: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.coordinate:
            raise ModelingError("step_guard_name_invalid")
        _nonnegative_int(self.tile, f"step_guard_tile_invalid:{self.name}", minimum=1)
        if not isinstance(self.lower_bound, int):
            raise ModelingError(f"step_guard_lower_bound_invalid:{self.name}")

    def step(self, value: Any) -> int:
        exact = as_fraction(value)
        if exact < self.lower_bound:
            raise ModelingError(
                f"guard_domain_miss:{self.name}:{self.coordinate}={exact}")
        return math.ceil(exact / self.tile)

    def select(self, value: Any) -> int:
        return self.step(value)

    def cell(self, index: int) -> GuardCell:
        _nonnegative_int(index, f"step_guard_index_invalid:{self.name}")
        if index == 0:
            if self.lower_bound > 0:
                raise ModelingError(f"step_guard_cell_empty:{self.name}:0")
            return GuardCell(self.coordinate, 0, None, 0, self.name)
        lower = Fraction((index - 1) * self.tile)
        upper = Fraction(index * self.tile)
        if index * self.tile < self.lower_bound:
            raise ModelingError(f"step_guard_cell_below_domain:{self.name}:{index}")
        return GuardCell(self.coordinate, index, lower, upper, self.name)

    def matches(self, value: Any, index: int) -> bool:
        try:
            cell = self.cell(index)
        except ModelingError:
            return False
        return cell.contains(value)

    def boundaries(self, index: int) -> tuple[Any, Any, Any]:
        """Return ``(kM-1, kM, kM+1)`` for tail boundary probes."""
        _nonnegative_int(index, f"step_guard_index_invalid:{self.name}", minimum=1)
        edge = index * self.tile
        return edge - 1, edge, edge + 1

    def verify(self, upper: int, *, lower: int | None = None) -> PartitionAudit:
        lower = self.lower_bound if lower is None else lower
        if lower < self.lower_bound or upper < lower:
            raise ModelingError(f"step_guard_audit_domain_invalid:{self.name}")
        max_index = self.step(upper)
        first = 0 if lower <= 0 else 1
        cells = tuple(self.cell(index) for index in range(first, max_index + 1))
        counts = [0] * (max_index + 1)
        misses: list[int] = []
        overlaps: list[int] = []
        for value in range(lower, upper + 1):
            hits = [cell.index for cell in cells if cell.contains(value)]
            if not hits:
                misses.append(value)
                continue
            if len(hits) > 1:
                overlaps.append(value)
            for index in hits:
                counts[index] += 1
        return PartitionAudit(
            self.coordinate, lower, upper, cells,
            tuple((index, count) for index, count in enumerate(counts) if count),
            tuple(misses), tuple(overlaps), upper - lower + 1)


@dataclass(frozen=True)
class CoreWaveGuard:
    """``ceil(J/C)`` waves with an explicit partially active tail wave."""

    cores: int
    name: str = "core_wave"
    coordinate: str = "tiles"

    def __post_init__(self) -> None:
        _nonnegative_int(self.cores, "core_wave_cores_invalid", minimum=1)

    def waves(self, tiles: Any) -> int:
        exact = as_fraction(tiles)
        if exact < 0:
            raise ModelingError(f"guard_domain_miss:{self.name}:{self.coordinate}")
        return math.ceil(exact / self.cores)

    def select(self, tiles: Any) -> int:
        return self.waves(tiles)

    def tail_cores(self, tiles: Any) -> int:
        total = int(as_fraction(tiles))
        if total <= 0:
            return 0
        return total - (self.waves(total) - 1) * self.cores

    def active_cores(self, tiles: Any) -> int:
        return min(self.cores, max(0, int(as_fraction(tiles))))

    def cell(self, index: int) -> GuardCell:
        _nonnegative_int(index, f"core_wave_index_invalid:{self.name}")
        if index == 0:
            return GuardCell(self.coordinate, 0, None, 0, self.name)
        return GuardCell(self.coordinate, index, Fraction((index - 1) * self.cores),
                         Fraction(index * self.cores), self.name)

    def matches(self, tiles: Any, index: int) -> bool:
        return self.cell(index).contains(tiles)

    def verify(self, upper: int, *, lower: int = 1) -> PartitionAudit:
        return StepGuard(self.name, self.coordinate, self.cores, 1).verify(
            upper, lower=lower)


@dataclass(frozen=True)
class SplitKPolicy:
    """Declared analytical split-K selection (never a vendor inference)."""

    target: Fraction | int
    max_splits: int = 8
    name: str = "split_k"

    def __post_init__(self) -> None:
        target = as_fraction(self.target)
        if target <= 0:
            raise ModelingError(f"split_k_target_invalid:{self.name}")
        object.__setattr__(self, "target", target)
        _nonnegative_int(self.max_splits, "split_k_max_splits_invalid", minimum=1)

    def splits(self, work: Any) -> int:
        exact = as_fraction(work)
        if exact < 0:
            raise ModelingError(f"guard_domain_miss:{self.name}:work")
        if exact == 0:
            return 1
        return max(1, min(self.max_splits, math.ceil(exact / self.target)))


@dataclass(frozen=True)
class SplitKGuard:
    """Piecewise split count driven by ``SplitKPolicy``."""

    policy: SplitKPolicy
    coordinate: str = "work"
    name: str = "split_k"

    def splits(self, work: Any) -> int:
        return self.policy.splits(work)

    def select(self, work: Any) -> int:
        return self.splits(work)

    def cell(self, index: int) -> GuardCell:
        _nonnegative_int(index, f"split_k_index_invalid:{self.name}")
        if index < 1 or index > self.policy.max_splits:
            raise ModelingError(f"split_k_index_out_of_range:{self.name}:{index}")
        lower = (None if index == 1
                 else Fraction((index - 1) * self.policy.target))
        upper = (None if index == self.policy.max_splits
                 else Fraction(index * self.policy.target))
        return GuardCell(self.coordinate, index, lower, upper, self.name)

    def matches(self, work: Any, index: int) -> bool:
        return self.cell(index).contains(work)

    def verify(self, upper: int, *, lower: int = 0) -> PartitionAudit:
        max_index = self.policy.max_splits
        cells = tuple(self.cell(index) for index in range(1, max_index + 1))
        counts = [0] * (max_index + 1)
        misses: list[int] = []
        overlaps: list[int] = []
        for value in range(lower, upper + 1):
            hits = [cell.index for cell in cells if cell.contains(value)]
            if not hits:
                misses.append(value)
                continue
            if len(hits) > 1:
                overlaps.append(value)
            for index in hits:
                counts[index] += 1
        return PartitionAudit(
            self.coordinate, lower, upper, cells,
            tuple((index, count) for index, count in enumerate(counts) if count),
            tuple(misses), tuple(overlaps), upper - lower + 1)


@dataclass(frozen=True)
class GraphBucketGuard:
    """``min{b in B : b >= T}`` graph capture bucket selection."""

    buckets: tuple[int, ...]
    name: str = "graph_bucket"
    coordinate: str = "T"

    def __post_init__(self) -> None:
        values = tuple(self.buckets)
        if not values or any(not isinstance(item, int) or isinstance(item, bool)
                             or item <= 0 for item in values):
            raise ModelingError(f"graph_bucket_invalid:{self.name}")
        if list(values) != sorted(set(values)):
            raise ModelingError(f"graph_bucket_not_strictly_ascending:{self.name}")
        object.__setattr__(self, "buckets", values)

    def select(self, value: Any) -> int:
        exact = as_fraction(value)
        for bucket in self.buckets:
            if exact <= bucket:
                return bucket
        raise ModelingError(f"guard_domain_miss:{self.name}:{self.coordinate}={exact}")

    def bucket(self, value: Any) -> int:
        return self.select(value)

    def select_index(self, value: Any) -> int:
        return self.buckets.index(self.select(value))

    def pad(self, value: Any) -> int:
        return self.select(value) - int(as_fraction(value))

    def cell(self, index: int) -> GuardCell:
        _nonnegative_int(index, f"graph_bucket_index_invalid:{self.name}")
        if index >= len(self.buckets):
            raise ModelingError(f"graph_bucket_index_out_of_range:{self.name}:{index}")
        lower = 0 if index == 0 else self.buckets[index - 1]
        return GuardCell(self.coordinate, index, Fraction(lower),
                         Fraction(self.buckets[index]), self.name)

    def matches(self, value: Any, index: int) -> bool:
        return self.cell(index).contains(value)

    def verify(self, *, lower: int = 1) -> PartitionAudit:
        return self._audit(lower, self.buckets[-1])

    def _audit(self, lower: int, upper: int) -> PartitionAudit:
        cells = tuple(self.cell(index) for index in range(len(self.buckets)))
        counts = [0] * len(cells)
        misses: list[int] = []
        overlaps: list[int] = []
        for value in range(lower, upper + 1):
            hits = [cell.index for cell in cells if cell.contains(value)]
            if not hits:
                misses.append(value)
                continue
            if len(hits) > 1:
                overlaps.append(value)
            for index in hits:
                counts[index] += 1
        return PartitionAudit(
            self.coordinate, lower, upper, cells,
            tuple((index, count) for index, count in enumerate(counts) if count),
            tuple(misses), tuple(overlaps), upper - lower + 1)


_Guard = StepGuard | CoreWaveGuard | SplitKGuard | GraphBucketGuard


def _guard_index(guard: _Guard, value: Any) -> int:
    if isinstance(guard, GraphBucketGuard):
        return guard.select_index(value)
    if isinstance(guard, StepGuard):
        return guard.select(value)
    if isinstance(guard, CoreWaveGuard):
        return guard.select(value)
    return guard.select(value)


@dataclass(frozen=True)
class StructuralPartition:
    """A conjunction of one-axis guards with a bounded unique-hit audit."""

    guards: tuple[_Guard, ...]
    name: str = "structural_partition"

    def __post_init__(self) -> None:
        if not self.guards:
            raise ModelingError(f"partition_guards_empty:{self.name}")
        object.__setattr__(self, "guards", tuple(self.guards))
        coordinates = [guard.coordinate for guard in self.guards]
        if len(set(coordinates)) != len(coordinates):
            raise ModelingError(f"partition_duplicate_coordinate:{self.name}")

    def select(self, env: Mapping[str, Any]) -> tuple[int, ...]:
        result = []
        for guard in self.guards:
            if guard.coordinate not in env:
                raise ModelingError(
                    f"partition_coordinate_missing:{self.name}:{guard.coordinate}")
            result.append(_guard_index(guard, env[guard.coordinate]))
        return tuple(result)

    def cell(self, indices: Sequence[int]) -> tuple[GuardCell, ...]:
        if len(indices) != len(self.guards):
            raise ModelingError(f"partition_index_arity:{self.name}")
        return tuple(guard.cell(index) for guard, index in zip(self.guards, indices))

    def verify(self, domains: Mapping[str, tuple[int, int]], *, limit: int = 200_000) -> "PartitionGridAudit":
        total = 1
        for guard in self.guards:
            if guard.coordinate not in domains:
                raise ModelingError(
                    f"partition_domain_missing:{self.name}:{guard.coordinate}")
            lower, upper = domains[guard.coordinate]
            total *= max(0, upper - lower + 1)
        if total > limit:
            raise ModelingError(f"partition_audit_budget_exceeded:{total}>{limit}")
        ranges = [range(domains[guard.coordinate][0], domains[guard.coordinate][1] + 1)
                  for guard in self.guards]
        domain_cells = [_guard_cells(guard, domains[guard.coordinate][1])
                        for guard in self.guards]
        misses: list[tuple[int, ...]] = []
        overlaps: list[tuple[int, ...]] = []
        hits = 0
        for point in _product(ranges):
            missing = False
            ambiguous = False
            for guard, cells, value in zip(self.guards, domain_cells, point):
                matched = [cell for cell in cells if cell.contains(value)]
                if not matched:
                    missing = True
                elif len(matched) > 1:
                    ambiguous = True
            if missing:
                misses.append(point)
            if ambiguous:
                overlaps.append(point)
            hits += 1
        return PartitionGridAudit(self.name, dict(domains), hits,
                                  tuple(misses), tuple(overlaps), total)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "guards": [{"name": guard.name, "coordinate": guard.coordinate}
                           for guard in self.guards]}


@dataclass(frozen=True)
class PartitionGridAudit:
    name: str
    domains: Mapping[str, tuple[int, int]]
    checked: int
    misses: tuple[tuple[int, ...], ...] = ()
    overlaps: tuple[tuple[int, ...], ...] = ()
    expected: int = 0

    @property
    def unique(self) -> bool:
        return not self.misses and not self.overlaps and self.checked == self.expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "domains": {key: list(value) for key, value
                                           in self.domains.items()},
            "checked": self.checked, "expected": self.expected,
            "misses": len(self.misses), "overlaps": len(self.overlaps),
            "unique": self.unique,
        }


def _guard_cells(guard: _Guard, upper: int) -> list[GuardCell]:
    if isinstance(guard, StepGuard):
        first = 0 if guard.lower_bound <= 0 else 1
        return [guard.cell(index) for index in range(first, guard.step(upper) + 1)]
    if isinstance(guard, CoreWaveGuard):
        return [guard.cell(index) for index in range(0, guard.waves(upper) + 1)]
    if isinstance(guard, SplitKGuard):
        return [guard.cell(index) for index in range(1, guard.policy.max_splits + 1)]
    if isinstance(guard, GraphBucketGuard):
        return [guard.cell(index) for index in range(len(guard.buckets))]
    raise ModelingError(f"guard_kind_unknown:{type(guard).__name__}")


def _product(ranges: Iterable[range]):
    if not ranges:
        yield ()
        return
    ranges = list(ranges)
    for value in ranges[0]:
        for tail in _product(ranges[1:]):
            yield (value, *tail)


@dataclass(frozen=True)
class PipeDemand:
    """Work on one physical pipe with an explicit native supply."""

    pipe: str
    work: Expr
    supply: Fraction | int
    unit: str = "work"

    def __post_init__(self) -> None:
        if not isinstance(self.pipe, str) or not self.pipe.strip():
            raise ModelingError("pipe_demand_name_invalid")
        if not isinstance(self.work, Expr):
            raise ModelingError(f"pipe_demand_work_invalid:{self.pipe}")
        supply = as_fraction(self.supply)
        if supply <= 0:
            raise ModelingError(f"pipe_demand_supply_invalid:{self.pipe}")
        object.__setattr__(self, "supply", supply)

    def time(self, env: Mapping[str, Any]) -> Fraction:
        return evaluate(self.work, env) / self.supply

    def to_dict(self) -> dict[str, Any]:
        return {"pipe": self.pipe, "work": str(self.work),
                "supply": str(self.supply), "unit": self.unit}


@dataclass(frozen=True)
class Crossover:
    """One continuous equality point between two declared pipe branches."""

    left: str
    right: str
    coordinate: str
    value: Fraction | float
    exact: bool
    degree: int
    dominance_before: str
    dominance_after: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left, "right": self.right,
            "coordinate": self.coordinate,
            "value": str(self.value), "exact": self.exact,
            "degree": self.degree,
            "dominance_before": self.dominance_before,
            "dominance_after": self.dominance_after,
        }


@dataclass(frozen=True)
class PipeDominanceGuard:
    """Continuous Cube/Vector/MTE2/MTE1/MTE3/Fixpipe/Link competition."""

    demands: tuple[PipeDemand, ...]
    name: str = "pipe_dominance"

    def __post_init__(self) -> None:
        demands = tuple(self.demands)
        if not demands:
            raise ModelingError(f"pipe_dominance_empty:{self.name}")
        pipes = [demand.pipe for demand in demands]
        if len(set(pipes)) != len(pipes):
            raise ModelingError(f"pipe_dominance_duplicate_pipe:{self.name}")
        object.__setattr__(self, "demands", demands)

    @property
    def pipes(self) -> tuple[str, ...]:
        return tuple(demand.pipe for demand in self.demands)

    def demand(self, pipe: str) -> PipeDemand:
        for demand in self.demands:
            if demand.pipe == pipe:
                return demand
        raise ModelingError(f"pipe_demand_missing:{pipe}")

    def times(self, env: Mapping[str, Any]) -> dict[str, Fraction]:
        return {demand.pipe: demand.time(env) for demand in self.demands}

    def envelope(self, env: Mapping[str, Any]) -> Fraction:
        return max(self.times(env).values())

    def dominant(self, env: Mapping[str, Any]) -> str:
        times = self.times(env)
        # Stable: max value, then declaration order via strict ``>`` scan.
        winner = self.demands[0].pipe
        for demand in self.demands[1:]:
            if times[demand.pipe] > times[winner]:
                winner = demand.pipe
        return winner

    def tied(self, env: Mapping[str, Any]) -> tuple[str, ...]:
        times = self.times(env)
        peak = max(times.values())
        return tuple(demand.pipe for demand in self.demands if times[demand.pipe] == peak)

    def crossover(self, left: str, right: str, coordinate: str, *,
                  at: Mapping[str, Any] | None = None,
                  lower: Fraction | int | None = None,
                  upper: Fraction | int | None = None) -> tuple[Crossover, ...]:
        """Exact roots of ``t_left == t_right`` in the coordinate.

        ``at`` binds every other symbolic value; coefficients must then be
        constant.  Unsupported non-polynomial branches raise.
        """
        left_demand, right_demand = self.demand(left), self.demand(right)
        difference = Sub(Div(left_demand.work, Const(left_demand.supply)),
                         Div(right_demand.work, Const(right_demand.supply)))
        polynomial = difference.polynomial(coordinate, degree=2)
        if polynomial is None:
            raise ModelingError(f"pipe_crossover_not_polynomial:{left}:{right}:{coordinate}")
        env = dict(at or {})
        if coordinate in env:
            raise ModelingError(f"pipe_crossover_coordinate_pinned:{coordinate}")
        coefficients = []
        for item in polynomial:
            unbound = item.free_symbols() - {coordinate} - set(env)
            if unbound:
                raise ModelingError(
                    f"pipe_crossover_unbound:{left}:{right}:{sorted(unbound)}")
            coefficients.append(evaluate(item, env))
        while len(coefficients) < 3:
            coefficients.append(Fraction(0))
        constant, linear, quadratic = coefficients[0], coefficients[1], coefficients[2]
        roots: list[tuple[Fraction | float, bool, int]] = []
        if quadratic != 0:
            degree = 2
            discriminant = linear * linear - 4 * quadratic * constant
            if discriminant < 0:
                roots = []
            elif discriminant == 0:
                roots = [(-linear / (2 * quadratic), True, degree)]
            else:
                exact_root = _exact_sqrt(discriminant)
                if exact_root is None:
                    square = math.sqrt(float(discriminant))
                    roots = [((-float(linear) - square) / (2 * float(quadratic)), False, degree),
                             ((-float(linear) + square) / (2 * float(quadratic)), False, degree)]
                else:
                    roots = [((-linear - exact_root) / (2 * quadratic), True, degree),
                             ((-linear + exact_root) / (2 * quadratic), True, degree)]
        elif linear != 0:
            degree = 1
            roots = [(-constant / linear, True, degree)]
        else:
            degree = 0
            roots = []
        lo = None if lower is None else as_fraction(lower)
        hi = None if upper is None else as_fraction(upper)
        filtered: list[tuple[Fraction | float, bool, int]] = []
        for root, exact, root_degree in roots:
            if lo is not None and root < lo:
                continue
            if hi is not None and root > hi:
                continue
            duplicate = False
            for existing, existing_exact, _ in filtered:
                if exact and existing_exact:
                    duplicate = root == existing
                elif not exact and not existing_exact:
                    duplicate = _close(root, existing)
                if duplicate:
                    break
            if not duplicate:
                filtered.append((root, exact, root_degree))
        filtered.sort(key=lambda item: item[0])
        result = []
        for position, (root, exact, root_degree) in enumerate(filtered):
            previous = filtered[position - 1][0] if position else None
            following = (filtered[position + 1][0]
                         if position + 1 < len(filtered) else None)
            before, after = self._dominance_around(
                left, right, coordinate, root, env, lo, hi,
                previous=previous, following=following)
            result.append(Crossover(left, right, coordinate, root, exact,
                                    root_degree, before, after))
        return tuple(result)

    def _dominance_around(self, left: str, right: str, coordinate: str,
                          root: Fraction | float, env: Mapping[str, Any],
                          lower: Fraction | None, upper: Fraction | None,
                          *, previous: Fraction | float | None = None,
                          following: Fraction | float | None = None
                          ) -> tuple[str, str]:
        delta = Fraction(1, 1000)
        exact_root = as_fraction(root)
        left_limit = (as_fraction(previous) if previous is not None else lower)
        right_limit = (as_fraction(following) if following is not None else upper)
        if left_limit is None:
            left_probe = exact_root - delta
        elif left_limit < exact_root:
            left_probe = (left_limit + exact_root) / 2
        else:
            left_probe = exact_root
        if right_limit is None:
            right_probe = exact_root + delta
        elif right_limit > exact_root:
            right_probe = (exact_root + right_limit) / 2
        else:
            right_probe = exact_root
        before_env = {**env, coordinate: left_probe}
        after_env = {**env, coordinate: right_probe}
        before = self.dominant(before_env)
        after = self.dominant(after_env)
        if before != left and before != right:
            before = self.dominant({**env, coordinate: exact_root})
        if after != left and after != right:
            after = self.dominant({**env, coordinate: exact_root})
        return before, after

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "demands": [demand.to_dict() for demand in self.demands]}


def _exact_sqrt(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator, denominator = value.numerator, value.denominator
    root_numerator = math.isqrt(numerator)
    root_denominator = math.isqrt(denominator)
    if root_numerator * root_numerator != numerator:
        return None
    if root_denominator * root_denominator != denominator:
        return None
    return Fraction(root_numerator, root_denominator)


def _close(left: Fraction | float, right: Fraction | float) -> bool:
    # Relative floating-point identity only.  A fixed coordinate tolerance
    # would merge physically distinct irrational roots whose separation is
    # small but resolvable (e.g. 1 +/- sqrt(2)*1e-8).
    a, b = float(left), float(right)
    return abs(a - b) <= 1e-15 * max(1.0, abs(a), abs(b))


@dataclass(frozen=True)
class ServiceMode:
    """A named service mode normalized onto one physical pipe."""

    name: str
    pipe: str
    work: Expr
    supply: Fraction | int
    unit: str = "work"

    def __post_init__(self) -> None:
        if not self.name or not self.pipe:
            raise ModelingError("service_mode_name_invalid")
        if not isinstance(self.work, Expr):
            raise ModelingError(f"service_mode_work_invalid:{self.name}")
        supply = as_fraction(self.supply)
        if supply <= 0:
            raise ModelingError(f"service_mode_supply_invalid:{self.name}")
        object.__setattr__(self, "supply", supply)

    def time(self, env: Mapping[str, Any]) -> Fraction:
        return evaluate(self.work, env) / self.supply


@dataclass(frozen=True)
class PipeEnvelope:
    """Alias-normalized envelope: sum per physical pipe, then max."""

    modes: tuple[ServiceMode, ...]
    name: str = "pipe_envelope"

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if not modes:
            raise ModelingError(f"pipe_envelope_empty:{self.name}")
        names = [mode.name for mode in modes]
        if len(set(names)) != len(names):
            raise ModelingError(f"pipe_envelope_duplicate_mode:{self.name}")
        object.__setattr__(self, "modes", modes)

    def pipe_times(self, env: Mapping[str, Any]) -> dict[str, Fraction]:
        result: dict[str, Fraction] = {}
        for mode in self.modes:
            result[mode.pipe] = result.get(mode.pipe, Fraction(0)) + mode.time(env)
        return result

    def duration(self, env: Mapping[str, Any]) -> Fraction:
        return max(self.pipe_times(env).values())

    def dominant(self, env: Mapping[str, Any]) -> str:
        times = self.pipe_times(env)
        winner = next(iter(times))
        for pipe, value in times.items():
            if value > times[winner]:
                winner = pipe
        return winner

    def isolated_duration(self, env: Mapping[str, Any]) -> Fraction:
        """The rejected model: one max per alias instead of a shared queue."""
        return max(mode.time(env) for mode in self.modes)


def tile_partition(name: str, coordinate: str, tile: int, upper: int) -> tuple[StepGuard, PartitionAudit]:
    guard = StepGuard(name, coordinate, tile)
    return guard, guard.verify(upper)
