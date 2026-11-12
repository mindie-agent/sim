"""Exact symbolic integer/rational expressions for piecewise modelling.

[Role]
  Provide one immutable expression algebra shared by structural work formulas
  and their numeric lowering.  ``Const/Var/Add/Sub/Mul/Div/Min/Max/FloorDiv/
  CeilDiv/Sum`` can be evaluated exactly with ``Fraction``, bounded with
  conservative intervals and inspected for their affine/quadratic structure.

[Invariants]
  * Evaluation is exact rational arithmetic; float bindings are read through
    their decimal spelling and never silently rounded by this module.
  * ``FloorDiv``/``CeilDiv`` require a strictly positive denominator when the
    denominator itself is symbolic.  An unbounded or zero-span denominator
    raises instead of returning an unsound bound.
  * Interval analysis is conservative: it may widen but never excludes a
    reachable value.  A missing variable raises ``expression_variable_unbound``.
  * Simplification is semantics-preserving and deterministic (sorted operands),
    so structurally equal expressions compare equal.
  * These expressions are continuous extensions of integer work formulas on
    the declared lattice, not an interpolation of measurements.

[Timeouts & Locks]
  Pure bounded tree recursion.  No iteration history, waits, I/O or locks.

[Side-Effect Policy]
  No filesystem/network/device mutation; every operation returns a new tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping

from mindie_sim.schema.common import ModelingError

def as_fraction(value: Any) -> Fraction:
    """Exact scalar conversion: ints and Fractions are exact, floats decimal."""
    t = type(value)
    if t is Fraction:
        return value
    if t is int:
        return Fraction(value)
    if t is bool or isinstance(value, bool):
        raise ModelingError(f"expression_non_numeric:{value!r}")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelingError(f"expression_non_finite:{value!r}")
        return Fraction(str(value))
    raise ModelingError(f"expression_non_numeric:{value!r}")

@dataclass(frozen=True)
class Interval:
    """A conservative closed interval; ``None`` denotes an open bound."""

    lower: Fraction | None
    upper: Fraction | None

    def __post_init__(self) -> None:
        if self.lower is not None:
            object.__setattr__(self, "lower", as_fraction(self.lower))
        if self.upper is not None:
            object.__setattr__(self, "upper", as_fraction(self.upper))
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ModelingError(f"interval_inverted:{self.lower}>:{self.upper}")

    @classmethod
    def point(cls, value: Any) -> "Interval":
        exact = as_fraction(value)
        return cls(exact, exact)

    @classmethod
    def nonnegative(cls) -> "Interval":
        return cls(Fraction(0), None)

    @property
    def bounded(self) -> bool:
        return self.lower is not None and self.upper is not None

    def contains(self, value: Any) -> bool:
        exact = as_fraction(value)
        return ((self.lower is None or exact >= self.lower)
                and (self.upper is None or exact <= self.upper))

    def width(self) -> Fraction | None:
        if not self.bounded:
            return None
        return self.upper - self.lower

    def intersect(self, other: "Interval") -> "Interval":
        lower = (other.lower if self.lower is None else self.lower
                 if other.lower is None else max(self.lower, other.lower))
        upper = (other.upper if self.upper is None else self.upper
                 if other.upper is None else min(self.upper, other.upper))
        return Interval(lower, upper)

    def __neg__(self) -> "Interval":
        return Interval(None if self.upper is None else -self.upper,
                        None if self.lower is None else -self.lower)

    def add(self, other: "Interval") -> "Interval":
        lower = None if self.lower is None or other.lower is None else self.lower + other.lower
        upper = None if self.upper is None or other.upper is None else self.upper + other.upper
        return Interval(lower, upper)

    def sub(self, other: "Interval") -> "Interval":
        return self.add(-other)

    def mul(self, other: "Interval") -> "Interval":
        if self.bounded and self.lower == self.upper == 0:
            return Interval.point(0)
        if other.bounded and other.lower == other.upper == 0:
            return Interval.point(0)
        if not (self.bounded and other.bounded):
            # A one-sided unbounded interval still has a usable nonnegative
            # product when both sides are known nonnegative.
            if (self.lower is not None and self.lower >= 0
                    and other.lower is not None and other.lower >= 0):
                lower = self.lower * other.lower
                upper = (None if self.upper is None or other.upper is None
                         else self.upper * other.upper)
                return Interval(lower, upper)
            return Interval(None, None)
        products = (self.lower * other.lower, self.lower * other.upper,
                    self.upper * other.lower, self.upper * other.upper)
        return Interval(min(products), max(products))

    def floor_div(self, other: "Interval") -> "Interval":
        if other.lower is None or other.upper is None or other.lower <= 0:
            raise ModelingError("interval_floor_div_requires_positive_denominator")
        if not self.bounded:
            return Interval(None, None)
        quotients = (self.lower / other.lower, self.lower / other.upper,
                     self.upper / other.lower, self.upper / other.upper)
        return Interval(Fraction(math.floor(min(quotients))),
                        Fraction(math.floor(max(quotients))))

    def ceil_div(self, other: "Interval") -> "Interval":
        if other.lower is None or other.upper is None or other.lower <= 0:
            raise ModelingError("interval_ceil_div_requires_positive_denominator")
        if not self.bounded:
            return Interval(None, None)
        quotients = (self.lower / other.lower, self.lower / other.upper,
                     self.upper / other.lower, self.upper / other.upper)
        return Interval(Fraction(math.ceil(min(quotients))),
                        Fraction(math.ceil(max(quotients))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": None if self.lower is None else str(self.lower),
            "upper": None if self.upper is None else str(self.upper),
        }

def _require_expression(value: Any, reason: str) -> "Expr":
    if not isinstance(value, Expr):
        raise ModelingError(reason)
    return value

def _expr_tuple(values: Any, reason: str, *, minimum: int = 0) -> tuple["Expr", ...]:
    if isinstance(values, (list, tuple)):
        result = tuple(values)
    else:
        result = (values,)
    if len(result) < minimum:
        raise ModelingError(reason)
    for value in result:
        _require_expression(value, reason)
    return result

class Expr:
    """Base class for the exact expression algebra."""

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        raise NotImplementedError

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        raise NotImplementedError

    @property
    def is_constant(self) -> bool:
        return not self.free_symbols()

    def free_symbols(self) -> frozenset[str]:
        raise NotImplementedError

    def simplify(self) -> "Expr":
        return simplify(self)

    def substitute(self, mapping: Mapping[str, "Expr | int | Fraction"]) -> "Expr":
        converted = {name: value if isinstance(value, Expr) else Const(value)
                     for name, value in mapping.items()}
        return substitute(self, converted)

    def polynomial(self, coordinate: str, *, degree: int = 2) -> tuple["Expr", ...] | None:
        from .analysis import polynomial
        return polynomial(self, coordinate, degree=degree)

    def affine(self, coordinate: str) -> tuple["Expr", "Expr"] | None:
        from .analysis import polynomial
        coefficients = polynomial(self, coordinate, degree=1)
        if coefficients is None or len(coefficients) > 2:
            return None
        while len(coefficients) < 2:
            coefficients = (*coefficients, Const(0))
        return coefficients[1], coefficients[0]

    def quadratic(self, coordinate: str) -> tuple["Expr", "Expr", "Expr"] | None:
        from .analysis import polynomial
        coefficients = polynomial(self, coordinate, degree=2)
        if coefficients is None or len(coefficients) > 3:
            return None
        while len(coefficients) < 3:
            coefficients = (*coefficients, Const(0))
        return coefficients[2], coefficients[1], coefficients[0]

    def __add__(self, other: Any) -> "Expr":
        return Add((self, _coerce(other)))

    def __radd__(self, other: Any) -> "Expr":
        return Add((_coerce(other), self))

    def __sub__(self, other: Any) -> "Expr":
        return Sub(self, _coerce(other))

    def __rsub__(self, other: Any) -> "Expr":
        return Sub(_coerce(other), self)

    def __mul__(self, other: Any) -> "Expr":
        return Mul((self, _coerce(other)))

    def __rmul__(self, other: Any) -> "Expr":
        return Mul((_coerce(other), self))

    def __truediv__(self, other: Any) -> "Expr":
        return Div(self, _coerce(other))

    def __rtruediv__(self, other: Any) -> "Expr":
        return Div(_coerce(other), self)

    def __neg__(self) -> "Expr":
        return Neg(self)

    def __str__(self) -> str:
        return _render(self)

    __repr__ = __str__

def _coerce(value: Any) -> Expr:
    return value if isinstance(value, Expr) else Const(value)

@dataclass(frozen=True, slots=True)
class Const(Expr):
    value: Fraction

    def __init__(self, value: Any = 0) -> None:
        object.__setattr__(self, "value", as_fraction(value))

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return self.value

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return Interval.point(self.value)

    def free_symbols(self) -> frozenset[str]:
        return frozenset()

@dataclass(frozen=True, slots=True)
class Var(Expr):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("expression_variable_name_invalid")

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        if self.name not in env:
            raise ModelingError(f"expression_variable_unbound:{self.name}")
        return as_fraction(env[self.name])

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        if self.name not in env:
            raise ModelingError(f"expression_variable_unbound:{self.name}")
        value = env[self.name]
        if not isinstance(value, Interval):
            raise ModelingError(f"expression_interval_required:{self.name}")
        return value

    def free_symbols(self) -> frozenset[str]:
        return frozenset({self.name})

@dataclass(frozen=True, slots=True)
class Neg(Expr):
    operand: Expr

    def __post_init__(self) -> None:
        _require_expression(self.operand, "expression_neg_operand_invalid")

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return -self.operand.evaluate(env)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return -self.operand.interval(env)

    def free_symbols(self) -> frozenset[str]:
        return self.operand.free_symbols()

@dataclass(frozen=True, slots=True)
class Add(Expr):
    terms: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms",
                           _expr_tuple(self.terms, "expression_add_terms_invalid",
                                       minimum=1))

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return sum((term.evaluate(env) for term in self.terms), Fraction(0))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        result = Interval.point(0)
        for term in self.terms:
            result = result.add(term.interval(env))
        return result

    def free_symbols(self) -> frozenset[str]:
        return frozenset().union(*(term.free_symbols() for term in self.terms))

@dataclass(frozen=True, slots=True)
class Sub(Expr):
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        _require_expression(self.left, "expression_sub_left_invalid")
        _require_expression(self.right, "expression_sub_right_invalid")

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return self.left.evaluate(env) - self.right.evaluate(env)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return self.left.interval(env).sub(self.right.interval(env))

    def free_symbols(self) -> frozenset[str]:
        return self.left.free_symbols() | self.right.free_symbols()

@dataclass(frozen=True, slots=True)
class Mul(Expr):
    factors: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors",
                           _expr_tuple(self.factors, "expression_mul_factors_invalid",
                                       minimum=1))

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        result = Fraction(1)
        for factor in self.factors:
            result *= factor.evaluate(env)
        return result

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        result = Interval.point(1)
        for factor in self.factors:
            result = result.mul(factor.interval(env))
        return result

    def free_symbols(self) -> frozenset[str]:
        return frozenset().union(*(factor.free_symbols() for factor in self.factors))

@dataclass(frozen=True, slots=True)
class Div(Expr):
    numerator: Expr
    denominator: Expr

    def __post_init__(self) -> None:
        _require_expression(self.numerator, "expression_div_numerator_invalid")
        _require_expression(self.denominator, "expression_div_denominator_invalid")

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        divisor = self.denominator.evaluate(env)
        if divisor == 0:
            raise ModelingError("expression_division_by_zero")
        return self.numerator.evaluate(env) / divisor

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        top = self.numerator.interval(env)
        bottom = self.denominator.interval(env)
        if bottom.lower is None or bottom.upper is None:
            raise ModelingError("interval_div_requires_bounded_denominator")
        if bottom.lower <= 0 <= bottom.upper:
            raise ModelingError("interval_div_denominator_straddles_zero")
        if not top.bounded:
            return Interval(None, None)
        quotients = (top.lower / bottom.lower, top.lower / bottom.upper,
                     top.upper / bottom.lower, top.upper / bottom.upper)
        return Interval(min(quotients), max(quotients))

    def free_symbols(self) -> frozenset[str]:
        return self.numerator.free_symbols() | self.denominator.free_symbols()

@dataclass(frozen=True, slots=True)
class FloorDiv(Expr):
    numerator: Expr
    denominator: Expr

    def __post_init__(self) -> None:
        _require_expression(self.numerator, "expression_floor_div_numerator_invalid")
        _require_expression(self.denominator, "expression_floor_div_denominator_invalid")

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        divisor = self.denominator.evaluate(env)
        if divisor == 0:
            raise ModelingError("expression_floor_div_by_zero")
        return Fraction(math.floor(self.numerator.evaluate(env) / divisor))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return self.numerator.interval(env).floor_div(self.denominator.interval(env))

    def free_symbols(self) -> frozenset[str]:
        return self.numerator.free_symbols() | self.denominator.free_symbols()

@dataclass(frozen=True, slots=True)
class CeilDiv(Expr):
    numerator: Expr
    denominator: Expr

    def __post_init__(self) -> None:
        _require_expression(self.numerator, "expression_ceil_div_numerator_invalid")
        _require_expression(self.denominator, "expression_ceil_div_denominator_invalid")

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        divisor = self.denominator.evaluate(env)
        if divisor == 0:
            raise ModelingError("expression_ceil_div_by_zero")
        return Fraction(math.ceil(self.numerator.evaluate(env) / divisor))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return self.numerator.interval(env).ceil_div(self.denominator.interval(env))

    def free_symbols(self) -> frozenset[str]:
        return self.numerator.free_symbols() | self.denominator.free_symbols()

@dataclass(frozen=True, slots=True)
class Min(Expr):
    terms: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms",
                           _expr_tuple(self.terms, "expression_min_terms_invalid",
                                       minimum=1))

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return min(term.evaluate(env) for term in self.terms)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        bounds = [term.interval(env) for term in self.terms]
        lowers = [item.lower for item in bounds]
        uppers = [item.upper for item in bounds if item.upper is not None]
        lower = None if any(value is None for value in lowers) else min(lowers)
        return Interval(lower, min(uppers) if uppers else None)

    def free_symbols(self) -> frozenset[str]:
        return frozenset().union(*(term.free_symbols() for term in self.terms))

@dataclass(frozen=True, slots=True)
class Max(Expr):
    terms: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms",
                           _expr_tuple(self.terms, "expression_max_terms_invalid",
                                       minimum=1))

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return max(term.evaluate(env) for term in self.terms)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        bounds = [term.interval(env) for term in self.terms]
        lowers = [item.lower for item in bounds if item.lower is not None]
        uppers = [item.upper for item in bounds]
        upper = None if any(value is None for value in uppers) else max(uppers)
        return Interval(max(lowers) if lowers else None, upper)

    def free_symbols(self) -> frozenset[str]:
        return frozenset().union(*(term.free_symbols() for term in self.terms))

@dataclass(frozen=True, slots=True)
class Sum(Expr):
    terms: tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms",
                           _expr_tuple(self.terms, "expression_sum_terms_invalid"))

    def evaluate(self, env: Mapping[str, Any]) -> Fraction:
        return sum((term.evaluate(env) for term in self.terms), Fraction(0))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        result = Interval.point(0)
        for term in self.terms:
            result = result.add(term.interval(env))
        return result

    def free_symbols(self) -> frozenset[str]:
        return frozenset().union(*(term.free_symbols() for term in self.terms))

# Evaluation and substitution

def evaluate(expr: Expr, env: Mapping[str, Any]) -> Fraction:
    return expr.evaluate(env)

def substitute(expr: Expr, mapping: Mapping[str, Expr]) -> Expr:
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Var):
        return mapping.get(expr.name, expr)
    if isinstance(expr, Neg):
        return Neg(substitute(expr.operand, mapping))
    if isinstance(expr, Add):
        return Add(tuple(substitute(term, mapping) for term in expr.terms))
    if isinstance(expr, Sub):
        return Sub(substitute(expr.left, mapping), substitute(expr.right, mapping))
    if isinstance(expr, Mul):
        return Mul(tuple(substitute(factor, mapping) for factor in expr.factors))
    if isinstance(expr, Div):
        return Div(substitute(expr.numerator, mapping),
                   substitute(expr.denominator, mapping))
    if isinstance(expr, FloorDiv):
        return FloorDiv(substitute(expr.numerator, mapping),
                        substitute(expr.denominator, mapping))
    if isinstance(expr, CeilDiv):
        return CeilDiv(substitute(expr.numerator, mapping),
                       substitute(expr.denominator, mapping))
    if isinstance(expr, Min):
        return Min(tuple(substitute(term, mapping) for term in expr.terms))
    if isinstance(expr, Max):
        return Max(tuple(substitute(term, mapping) for term in expr.terms))
    if isinstance(expr, Sum):
        return Sum(tuple(substitute(term, mapping) for term in expr.terms))
    raise ModelingError(f"expression_node_unknown:{type(expr).__name__}")

# Symbolic simplification

def _sort_key(expr: Expr) -> tuple[str, str]:
    return type(expr).__name__, str(expr)

def _term_parts(expr: Expr) -> tuple[Fraction, tuple[Expr, ...]]:
    """Split ``c * x1 * ... * xn`` into coefficient and non-constant factors."""
    if isinstance(expr, Const):
        return expr.value, ()
    if isinstance(expr, Neg):
        coefficient, factors = _term_parts(expr.operand)
        return -coefficient, factors
    if isinstance(expr, Mul):
        coefficient = Fraction(1)
        factors: list[Expr] = []
        for factor in expr.factors:
            inner_coefficient, inner_factors = _term_parts(factor)
            coefficient *= inner_coefficient
            factors.extend(inner_factors)
        return coefficient, tuple(factors)
    if isinstance(expr, Div) and isinstance(expr.denominator, Const):
        coefficient, factors = _term_parts(expr.numerator)
        return coefficient / expr.denominator.value, factors
    return Fraction(1), (expr,)

def _has_guarded_denominator(expr: Expr) -> bool:
    """True when evaluation can raise through a non-constant denominator."""
    if isinstance(expr, (Div, FloorDiv, CeilDiv)):
        if not isinstance(expr.denominator, Const):
            return True
        return (_has_guarded_denominator(expr.denominator)
                or _has_guarded_denominator(expr.numerator))
    if isinstance(expr, Neg):
        return _has_guarded_denominator(expr.operand)
    if isinstance(expr, Sub):
        return (_has_guarded_denominator(expr.left)
                or _has_guarded_denominator(expr.right))
    if isinstance(expr, (Add, Mul, Min, Max, Sum)):
        items = expr.factors if isinstance(expr, Mul) else expr.terms
        return any(_has_guarded_denominator(item) for item in items)
    return False

def _build_product(coefficient: Fraction, factors: tuple[Expr, ...]) -> Expr:
    ordered = tuple(sorted(factors, key=_sort_key))
    if coefficient == 0:
        # ``0 * (1/x)`` is zero only where ``x`` is defined.  Keep the guarded
        # factor so simplification cannot erase the denominator obligation.
        if any(_has_guarded_denominator(factor) for factor in ordered):
            return Mul((Const(0), *ordered))
        return Const(0)
    if not ordered:
        return Const(coefficient)
    if coefficient == 1:
        return ordered[0] if len(ordered) == 1 else Mul(ordered)
    return Mul((Const(coefficient), *ordered))

def _simplify_add(terms: Iterable[Expr]) -> Expr:
    flattened: list[Expr] = []
    for term in terms:
        term = simplify(term)
        if isinstance(term, Add):
            flattened.extend(term.terms)
        elif isinstance(term, Sum):
            flattened.extend(term.terms)
        else:
            flattened.append(term)
    constant = Fraction(0)
    grouped: dict[tuple[Expr, ...], Fraction] = {}
    for term in flattened:
        coefficient, factors = _term_parts(term)
        if not factors:
            constant += coefficient
        else:
            grouped[tuple(sorted(factors, key=_sort_key))] = (
                grouped.get(tuple(sorted(factors, key=_sort_key)), Fraction(0))
                + coefficient)
    result: list[Expr] = []
    if constant:
        result.append(Const(constant))
    for factors, coefficient in sorted(grouped.items(),
                                       key=lambda item: tuple(map(_sort_key, item[0]))):
        if coefficient:
            result.append(_build_product(coefficient, factors))
        elif any(_has_guarded_denominator(factor) for factor in factors):
            # Cancelled terms still carry their denominator obligations.
            result.append(_build_product(0, factors))
    if not result:
        return Const(0)
    if len(result) == 1:
        return result[0]
    return Add(tuple(sorted(result, key=_sort_key)))

def _simplify_mul(factors: Iterable[Expr]) -> Expr:
    flattened: list[Expr] = []
    for factor in factors:
        factor = simplify(factor)
        if isinstance(factor, Mul):
            flattened.extend(factor.factors)
        else:
            flattened.append(factor)
    coefficient = Fraction(1)
    remaining: list[Expr] = []
    for factor in flattened:
        if isinstance(factor, Const):
            coefficient *= factor.value
        else:
            remaining.append(factor)
    if coefficient == 0:
        if any(_has_guarded_denominator(factor) for factor in remaining):
            remaining.sort(key=_sort_key)
            return Mul((Const(0), *remaining))
        return Const(0)
    if not remaining:
        return Const(coefficient)
    if coefficient != 1:
        remaining.append(Const(coefficient))
    remaining.sort(key=_sort_key)
    if len(remaining) == 1:
        return remaining[0]
    return Mul(tuple(remaining))

def simplify(expr: Expr) -> Expr:
    """Deterministic semantics-preserving normalisation."""
    if isinstance(expr, (Const, Var)):
        return expr
    if isinstance(expr, Neg):
        operand = simplify(expr.operand)
        if isinstance(operand, Const):
            return Const(-operand.value)
        if isinstance(operand, Neg):
            return operand.operand
        return Neg(operand)
    if isinstance(expr, Sub):
        return _simplify_add((expr.left, Neg(expr.right)))
    if isinstance(expr, Add):
        return _simplify_add(expr.terms)
    if isinstance(expr, Mul):
        return _simplify_mul(expr.factors)
    if isinstance(expr, Div):
        numerator, denominator = simplify(expr.numerator), simplify(expr.denominator)
        if isinstance(denominator, Const):
            if denominator.value == 0:
                raise ModelingError("expression_division_by_zero")
            if isinstance(numerator, Const):
                return Const(numerator.value / denominator.value)
            if denominator.value == 1:
                return numerator
            if isinstance(numerator, Mul):
                coefficient, factors = _term_parts(numerator)
                return _build_product(coefficient / denominator.value, factors)
        return Div(numerator, denominator)
    if isinstance(expr, FloorDiv):
        numerator, denominator = simplify(expr.numerator), simplify(expr.denominator)
        if isinstance(denominator, Const):
            if denominator.value <= 0:
                raise ModelingError("expression_floor_div_requires_positive_denominator")
            if isinstance(numerator, Const):
                return Const(math.floor(numerator.value / denominator.value))
        return FloorDiv(numerator, denominator)
    if isinstance(expr, CeilDiv):
        numerator, denominator = simplify(expr.numerator), simplify(expr.denominator)
        if isinstance(denominator, Const):
            if denominator.value <= 0:
                raise ModelingError("expression_ceil_div_requires_positive_denominator")
            if isinstance(numerator, Const):
                return Const(math.ceil(numerator.value / denominator.value))
        return CeilDiv(numerator, denominator)
    if isinstance(expr, (Min, Max)):
        kind = type(expr)
        flattened: list[Expr] = []
        for term in expr.terms:
            term = simplify(term)
            if isinstance(term, kind):
                flattened.extend(term.terms)
            else:
                flattened.append(term)
        unique = {str(term): term for term in flattened}
        items = list(unique.values())
        constants = [term.value for term in items if isinstance(term, Const)]
        if constants:
            selected = min(constants) if kind is Min else max(constants)
            items = [term for term in items
                     if not isinstance(term, Const) or term.value == selected]
            dropped = [term for term in items if isinstance(term, Const)]
            items = [term for term in items if not isinstance(term, Const)]
            items.append(Const(selected))
            del dropped
        items.sort(key=_sort_key)
        if len(items) == 1:
            return items[0]
        return kind(tuple(items))
    if isinstance(expr, Sum):
        flattened = []
        constant = Fraction(0)
        for term in expr.terms:
            term = simplify(term)
            if isinstance(term, Sum):
                flattened.extend(term.terms)
            elif isinstance(term, Const):
                constant += term.value
            else:
                flattened.append(term)
        if constant:
            flattened.append(Const(constant))
        if not flattened:
            return Const(0)
        if len(flattened) == 1:
            return flattened[0]
        return Sum(tuple(sorted(flattened, key=_sort_key)))
    raise ModelingError(f"expression_node_unknown:{type(expr).__name__}")

def _render(expr: Expr) -> str:
    if isinstance(expr, Const):
        if expr.value.denominator == 1:
            return str(expr.value.numerator)
        return f"{expr.value.numerator}/{expr.value.denominator}"
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Neg):
        return f"-({_render(expr.operand)})"
    if isinstance(expr, Add):
        return "(" + " + ".join(_render(term) for term in expr.terms) + ")"
    if isinstance(expr, Sub):
        return f"({_render(expr.left)} - {_render(expr.right)})"
    if isinstance(expr, Mul):
        return "(" + " * ".join(_render(factor) for factor in expr.factors) + ")"
    if isinstance(expr, Div):
        return f"({_render(expr.numerator)} / {_render(expr.denominator)})"
    if isinstance(expr, FloorDiv):
        return f"floor({_render(expr.numerator)} / {_render(expr.denominator)})"
    if isinstance(expr, CeilDiv):
        return f"ceil({_render(expr.numerator)} / {_render(expr.denominator)})"
    if isinstance(expr, Min):
        return "min(" + ", ".join(_render(term) for term in expr.terms) + ")"
    if isinstance(expr, Max):
        return "max(" + ", ".join(_render(term) for term in expr.terms) + ")"
    if isinstance(expr, Sum):
        return "sum(" + ", ".join(_render(term) for term in expr.terms) + ")"
    raise ModelingError(f"expression_node_unknown:{type(expr).__name__}")

def variables(*names: str) -> tuple[Var, ...]:
    return tuple(Var(name) for name in names)

def total(terms: Iterable[Expr]) -> Expr:
    return Sum(tuple(terms))
