"""Compiled exact integer-pair evaluation for ``Expr`` trees.

[Role]
  Lower immutable ``Expr`` AST nodes once into exact ``(numerator, denominator)``
  integer closures (``denominator >= 1``) so repeated operator/layer evaluations
  run in raw C-level Python ``int`` arithmetic without intermediate ``Fraction``
  allocations or redundant ``math.gcd`` reductions when ``denominator == 1``.

[Invariants]
  * Every compiled node preserves exact rational semantics, operand evaluation
    order (including denominator-first checks on ``Div``/``FloorDiv``/``CeilDiv``),
    and exact ``ModelingError`` codes from ``expressions.py``.
  * ``(numerator, denominator)`` always maintains ``denominator >= 1``; non-unit
    denominators stay reduced by ``math.gcd``.

[Timeouts & Locks]
  Pure bounded in-memory tree compilation and integer arithmetic; no I/O or locks.

[Side-Effect Policy]
  Process-local pure function memo on immutable ``Expr`` nodes only; never mutates
  caller environments or expression trees.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
from typing import Any, Callable, Mapping

from mindie_sim.schema.common import ModelingError

EvalPair = Callable[[Mapping[str, Any]], tuple[int, int]]


def _env_pair(raw: Any) -> tuple[int, int]:
    cls = type(raw)
    if cls is int:
        return raw, 1
    if cls is Fraction:
        return raw.numerator, raw.denominator
    if cls is bool or isinstance(raw, bool):
        return int(raw), 1
    if isinstance(raw, Fraction):
        return raw.numerator, raw.denominator
    if isinstance(raw, int):
        return int(raw), 1
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ModelingError(f"expression_non_finite:{raw!r}")
        exact = Fraction(str(raw))
        return exact.numerator, exact.denominator
    raise ModelingError(f"expression_non_numeric:{raw!r}")


def _reduce(num: int, den: int) -> tuple[int, int]:
    if den < 0:
        num, den = -num, -den
    if den == 1:
        return num, 1
    if num == 0:
        return 0, 1
    divisor = math.gcd(num, den)
    return num // divisor, den // divisor


def _add_rat(n0: int, d0: int, n1: int, d1: int) -> tuple[int, int]:
    if d0 == d1:
        return _reduce(n0 + n1, d0)
    return _reduce(n0 * d1 + n1 * d0, d0 * d1)


def _mul_rat(n0: int, d0: int, n1: int, d1: int) -> tuple[int, int]:
    return _reduce(n0 * n1, d0 * d1)


def _compile_node(expr: Any) -> EvalPair:
    from .expressions import (
        Add, CeilDiv, Const, Div, Expr, FloorDiv, Max, Min, Mul, Neg, Sub, Sum, Var,
    )

    cls = type(expr)
    if cls is Const:
        cn, cd = expr.value.numerator, expr.value.denominator
        return lambda _env, n=cn, d=cd: (n, d)
    if cls is Var:
        name = expr.name

        def _eval_var(env: Mapping[str, Any]) -> tuple[int, int]:
            if name not in env:
                raise ModelingError(f"expression_variable_unbound:{name}")
            val = env[name]
            return (val, 1) if type(val) is int else _env_pair(val)

        return _eval_var
    if cls is Neg:
        inner = _compile_cached(expr.operand)

        def _eval_neg(env: Mapping[str, Any]) -> tuple[int, int]:
            num, den = inner(env)
            return -num, den

        return _eval_neg
    if cls is Sub:
        lfn, rfn = _compile_cached(expr.left), _compile_cached(expr.right)

        def _eval_sub(env: Mapping[str, Any]) -> tuple[int, int]:
            ln, ld = lfn(env)
            rn, rd = rfn(env)
            return (ln - rn, 1) if ld == 1 and rd == 1 else _add_rat(ln, ld, -rn, rd)

        return _eval_sub
    if cls is Add or cls is Sum:
        fns = tuple(_compile_cached(term) for term in expr.terms)
        if not fns:
            return lambda _env: (0, 1)
        if len(fns) == 1:
            return fns[0]

        def _eval_sum(env: Mapping[str, Any]) -> tuple[int, int]:
            acc_n, acc_d = 0, 1
            for fn in fns:
                n, d = fn(env)
                if acc_d == 1 and d == 1:
                    acc_n += n
                else:
                    acc_n, acc_d = _add_rat(acc_n, acc_d, n, d)
            return acc_n, acc_d

        return _eval_sum
    if cls is Mul:
        fns = tuple(_compile_cached(factor) for factor in expr.factors)
        if len(fns) == 1:
            return fns[0]

        def _eval_mul(env: Mapping[str, Any]) -> tuple[int, int]:
            acc_n, acc_d = 1, 1
            for fn in fns:
                n, d = fn(env)
                if acc_d == 1 and d == 1:
                    acc_n *= n
                else:
                    acc_n, acc_d = _mul_rat(acc_n, acc_d, n, d)
            return acc_n, acc_d

        return _eval_mul
    if cls is Div:
        nfn, dfn = _compile_cached(expr.numerator), _compile_cached(expr.denominator)

        def _eval_div(env: Mapping[str, Any]) -> tuple[int, int]:
            dn, dd = dfn(env)
            if dn == 0:
                raise ModelingError("expression_division_by_zero")
            nn, nd = nfn(env)
            return _reduce(nn * dd, nd * dn)

        return _eval_div
    if cls is FloorDiv:
        nfn, dfn = _compile_cached(expr.numerator), _compile_cached(expr.denominator)

        def _eval_fdiv(env: Mapping[str, Any]) -> tuple[int, int]:
            dn, dd = dfn(env)
            if dn == 0:
                raise ModelingError("expression_floor_div_by_zero")
            nn, nd = nfn(env)
            if nd == 1 and dd == 1:
                return (nn // dn if dn > 0 else (-nn) // (-dn)), 1
            p, q = nn * dd, nd * dn
            if q < 0:
                p, q = -p, -q
            return p // q, 1

        return _eval_fdiv
    if cls is CeilDiv:
        nfn, dfn = _compile_cached(expr.numerator), _compile_cached(expr.denominator)

        def _eval_cdiv(env: Mapping[str, Any]) -> tuple[int, int]:
            dn, dd = dfn(env)
            if dn == 0:
                raise ModelingError("expression_ceil_div_by_zero")
            nn, nd = nfn(env)
            if nd == 1 and dd == 1:
                return ((nn + dn - 1) // dn if dn > 0 else (-nn - dn - 1) // (-dn)), 1
            p, q = nn * dd, nd * dn
            if q < 0:
                p, q = -p, -q
            return (p + q - 1) // q, 1

        return _eval_cdiv
    if cls is Min or cls is Max:
        fns = tuple(_compile_cached(term) for term in expr.terms)
        if len(fns) == 1:
            return fns[0]
        pick_min = cls is Min

        def _eval_ext(env: Mapping[str, Any]) -> tuple[int, int]:
            bn, bd = fns[0](env)
            for fn in fns[1:]:
                n, d = fn(env)
                less = (n < bn) if d == bd else (n * bd < bn * d)
                if less if pick_min else (not less and (n != bn or d != bd)):
                    bn, bd = n, d
            return bn, bd

        return _eval_ext
    if isinstance(expr, Expr):

        def _eval_fallback(env: Mapping[str, Any]) -> tuple[int, int]:
            res = expr._slow_evaluate(env)
            return res.numerator, res.denominator

        return _eval_fallback
    raise ModelingError(f"expression_node_unknown:{cls.__name__}")


@lru_cache(None)
def _compile_cached(expr: Any) -> EvalPair:
    return _compile_node(expr)


_FROM_COPRIME = getattr(Fraction, "_from_coprime_ints", None)


def evaluate_fast(expr: Any, env: Mapping[str, Any]) -> Fraction:
    try:
        fn = _compile_cached(expr)
    except TypeError:
        fn = _compile_node(expr)
    num, den = fn(env)
    if den == 1:
        return Fraction(num)
    if _FROM_COPRIME is not None:
        return _FROM_COPRIME(num, den)
    return Fraction(num, den)
