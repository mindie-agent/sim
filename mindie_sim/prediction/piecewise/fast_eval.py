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
_ZERO = 0
_ONE = 1
_ZERO_PAIR = (_ZERO, _ONE)


def _env_pair(raw: Any) -> tuple[int, int]:
    cls = type(raw)
    if cls is int:
        return raw, _ONE
    if cls is Fraction:
        return raw.numerator, raw.denominator
    if cls is bool or isinstance(raw, bool):
        return int(raw), _ONE
    if isinstance(raw, Fraction):
        return raw.numerator, raw.denominator
    if isinstance(raw, int):
        return int(raw), _ONE
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ModelingError(f"expression_non_finite:{raw!r}")
        exact = Fraction(str(raw))
        return exact.numerator, exact.denominator
    raise ModelingError(f"expression_non_numeric:{raw!r}")


def _reduce(num: int, den: int) -> tuple[int, int]:
    zi = _ZERO
    oi = _ONE
    if den < zi:
        num, den = -num, -den
    if den == oi:
        return num, oi
    if num == zi:
        return _ZERO_PAIR
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

    zi = _ZERO
    oi = _ONE
    zpair = _ZERO_PAIR
    cls = type(expr)
    if cls is Const:
        cn, cd = expr.value.numerator, expr.value.denominator
        return lambda _env, n=cn, d=cd: (n, d)
    if cls is Var:
        name = expr.name

        def _eval_var(env: Mapping[str, Any], u: int = oi) -> tuple[int, int]:
            if name not in env:
                raise ModelingError(f"expression_variable_unbound:{name}")
            val = env[name]
            return (val, u) if type(val) is int else _env_pair(val)

        return _eval_var
    if cls is Neg:
        inner = _compile_cached(expr.operand)

        def _eval_neg(env: Mapping[str, Any]) -> tuple[int, int]:
            num, den = inner(env)
            return -num, den

        return _eval_neg
    if cls is Sub:
        lfn, rfn = _compile_cached(expr.left), _compile_cached(expr.right)

        def _eval_sub(env: Mapping[str, Any], u: int = oi) -> tuple[int, int]:
            ln, ld = lfn(env)
            rn, rd = rfn(env)
            return (ln - rn, u) if ld == u and rd == u else _add_rat(ln, ld, -rn, rd)

        return _eval_sub
    if cls is Add or cls is Sum:
        fns = tuple(_compile_cached(term) for term in expr.terms)
        if not fns:
            return lambda _env, zp=zpair: zp
        if len(fns) == oi:
            return fns[zi]

        def _eval_sum(env: Mapping[str, Any], z: int = zi, u: int = oi) -> tuple[int, int]:
            acc_n = z
            acc_d = u
            for fn in fns:
                n, d = fn(env)
                if acc_d == u and d == u:
                    acc_n += n
                else:
                    acc_n, acc_d = _add_rat(acc_n, acc_d, n, d)
            return acc_n, acc_d

        return _eval_sum
    if cls is Mul:
        fns = tuple(_compile_cached(factor) for factor in expr.factors)
        if len(fns) == oi:
            return fns[zi]

        def _eval_mul(env: Mapping[str, Any], u: int = oi) -> tuple[int, int]:
            acc_n = u
            acc_d = u
            for fn in fns:
                n, d = fn(env)
                if acc_d == u and d == u:
                    acc_n *= n
                else:
                    acc_n, acc_d = _mul_rat(acc_n, acc_d, n, d)
            return acc_n, acc_d

        return _eval_mul
    if cls is Div:
        nfn, dfn = _compile_cached(expr.numerator), _compile_cached(expr.denominator)

        def _eval_div(env: Mapping[str, Any], z: int = zi) -> tuple[int, int]:
            dn, dd = dfn(env)
            if dn == z:
                raise ModelingError("expression_division_by_zero")
            nn, nd = nfn(env)
            return _reduce(nn * dd, nd * dn)

        return _eval_div
    if cls is FloorDiv:
        nfn, dfn = _compile_cached(expr.numerator), _compile_cached(expr.denominator)

        def _eval_fdiv(env: Mapping[str, Any], z: int = zi, u: int = oi) -> tuple[int, int]:
            dn, dd = dfn(env)
            if dn == z:
                raise ModelingError("expression_floor_div_by_zero")
            nn, nd = nfn(env)
            if nd == u and dd == u:
                return (nn // dn if dn > z else (-nn) // (-dn)), u
            p, q = nn * dd, nd * dn
            if q < z:
                p, q = -p, -q
            return p // q, u

        return _eval_fdiv
    if cls is CeilDiv:
        nfn, dfn = _compile_cached(expr.numerator), _compile_cached(expr.denominator)

        def _eval_cdiv(env: Mapping[str, Any], z: int = zi, u: int = oi) -> tuple[int, int]:
            dn, dd = dfn(env)
            if dn == z:
                raise ModelingError("expression_ceil_div_by_zero")
            nn, nd = nfn(env)
            if nd == u and dd == u:
                return ((nn + dn - u) // dn if dn > z else (-nn - dn - u) // (-dn)), u
            p, q = nn * dd, nd * dn
            if q < z:
                p, q = -p, -q
            return (p + q - u) // q, u

        return _eval_cdiv
    if cls is Min or cls is Max:
        fns = tuple(_compile_cached(term) for term in expr.terms)
        if len(fns) == oi:
            return fns[zi]
        pick_min = cls is Min
        head_fn = fns[zi]
        tail_fns = fns[oi:]

        def _eval_ext(env: Mapping[str, Any]) -> tuple[int, int]:
            bn, bd = head_fn(env)
            for fn in tail_fns:
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


_MAX_COMPILED_CACHE = 4096


@lru_cache(maxsize=_MAX_COMPILED_CACHE)
def _compile_cached(expr: Any) -> EvalPair:
    return _compile_node(expr)


_FROM_COPRIME = getattr(Fraction, "_from_coprime_ints", None)


def evaluate_fast(expr: Any, env: Mapping[str, Any]) -> Fraction:
    try:
        fn = _compile_cached(expr)
    except TypeError:
        fn = _compile_node(expr)
    num, den = fn(env)
    if den == _ONE:
        return Fraction(num)
    if _FROM_COPRIME is not None:
        return _FROM_COPRIME(num, den)
    return Fraction(num, den)
