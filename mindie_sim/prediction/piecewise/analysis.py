"""Polynomial inspection and crossover coefficient extraction.

[Role] Extract ascending coefficient expressions for affine/quadratic
  analysis used by continuous pipe-competition guards.
[Invariants] Only declared polynomial structure is accepted; an unsupported
  FloorDiv/Min/Max/non-constant denominator returns ``None`` instead of a
  fabricated derivative.  Coefficients are simplified expressions.
[Timeouts & Locks] Bounded tree recursion; no I/O or locks.
[Side-Effect Policy] Pure expression construction.
"""
from __future__ import annotations

from .expressions import Add, Const, Div, Expr, Mul, Neg, Sub, Sum, Var, simplify


def polynomial(expr: Expr, coordinate: str, *, degree: int = 2) -> tuple[Expr, ...] | None:
    """Return ascending coefficient expressions, or ``None`` when unsupported."""
    result = _polynomial(expr, coordinate, degree=degree)
    if result is None:
        return None
    return tuple(simplify(item) for item in result)


def _polynomial(expr: Expr, coordinate: str, *, degree: int = 2) -> tuple[Expr, ...] | None:
    if degree < 0:
        raise ModelingError("expression_polynomial_degree_invalid")
    if coordinate not in expr.free_symbols():
        return (expr,)
    if isinstance(expr, Var):
        return (Const(0), Const(1)) if expr.name == coordinate else (expr,)
    if isinstance(expr, Const):
        return (expr,)
    if isinstance(expr, Neg):
        inner = _polynomial(expr.operand, coordinate, degree=degree)
        return None if inner is None else tuple(Neg(item) for item in inner)
    if isinstance(expr, Sub):
        left = _polynomial(expr.left, coordinate, degree=degree)
        right = _polynomial(expr.right, coordinate, degree=degree)
        if left is None or right is None:
            return None
        return _poly_add(left, tuple(Neg(item) for item in right))
    if isinstance(expr, Add):
        result: tuple[Expr, ...] = (Const(0),)
        for term in expr.terms:
            inner = _polynomial(term, coordinate, degree=degree)
            if inner is None:
                return None
            result = _poly_add(result, inner)
        return result
    if isinstance(expr, Mul):
        result: tuple[Expr, ...] = (Const(1),)
        for factor in expr.factors:
            inner = _polynomial(factor, coordinate, degree=degree)
            if inner is None:
                return None
            if len(result) + len(inner) - 1 > degree + 1:
                return None
            product = [Const(0) for _ in range(len(result) + len(inner) - 1)]
            for left_index, left in enumerate(result):
                for right_index, right in enumerate(inner):
                    product[left_index + right_index] = Add(
                        (product[left_index + right_index], Mul((left, right))))
            result = tuple(product)
        return result
    if isinstance(expr, Div):
        if coordinate in expr.denominator.free_symbols():
            return None
        inner = _polynomial(expr.numerator, coordinate, degree=degree)
        if inner is None:
            return None
        return tuple(Div(item, expr.denominator) for item in inner)
    if isinstance(expr, Sum):
        result = (Const(0),)
        for term in expr.terms:
            inner = _polynomial(term, coordinate, degree=degree)
            if inner is None:
                return None
            result = _poly_add(result, inner)
        return result
    return None


def _poly_add(left: tuple[Expr, ...], right: tuple[Expr, ...]) -> tuple[Expr, ...]:
    size = max(len(left), len(right))
    result = []
    for index in range(size):
        terms = []
        if index < len(left):
            terms.append(left[index])
        if index < len(right):
            terms.append(right[index])
        result.append(simplify(Add(tuple(terms))))
    while len(result) > 1 and isinstance(result[-1], Const) and result[-1].value == 0:
        result.pop()
    return tuple(result)


# ---------------------------------------------------------------------------
# Rendering helpers
