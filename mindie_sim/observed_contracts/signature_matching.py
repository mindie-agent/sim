"""BIND1 declared/observed shape and dtype alignment helpers.

[Role]
  Match parsed observed lanes against declared ``typed`` role shapes and
  dtypes, returning True/False for a decidable comparison and None when slot
  ordering or an untyped role leaves the comparison ambiguous.

[Invariants]
  * Ordered role specs match only when every observed slot agrees; a tail role
    may consume one leading contiguous dimension.
  * Missing declared dtypes degrade to ambiguity (None), never to a silent
    positive match.
  * Counters compare multiset containment without mutating either operand.

[Timeouts & Locks]
  Pure in-memory tuple and counter inspection; no waits or locks.

[Side-Effect Policy]
  No I/O and no mutation of the caller's mappings, counters or shapes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .signature_vocabulary import (
    _ATTENTION_AUX_DTYPES,
    _AUX_ONLY_ROLES,
    _COARSE_SHAPE_KEYS,
    _DECLARED_INPUT_DTYPE_ROLES,
    _INPUT_ROLE_ORDER,
    _OUTPUT_SHAPE_KEYS,
    _PARTIAL_CACHE_ROLES,
    _PRIMARY_INPUT_KEYS,
    _ROLE_DTYPE_KEYS,
    _TAIL_INPUT_ROLES,
    _norm_dtype,
)


def _counter_leq(left: Counter[Any], right: Counter[Any]) -> bool:
    return all(left[key] <= right[key] for key in left)


def _as_shape_tuple(value: Any) -> tuple[int, ...] | None:
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return tuple(value)
    return None


def _lookup_dtype(typed: Mapping[str, Any], attrs: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = typed.get(key)
        if raw is None:
            raw = attrs.get(key)
        if (dt := _norm_dtype(raw)) is not None:
            return dt
    return None


def _parse_declared_shapes(
    shapes_field: Any,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    tuple[tuple[int, ...], ...] | None,
    tuple[tuple[int, ...], ...] | None,
] | None:
    if not isinstance(shapes_field, Mapping) or not shapes_field:
        return None
    coarse: dict[str, tuple[int, ...]] = {}
    in_roles: dict[str, tuple[int, ...]] = {}
    out_roles: dict[str, tuple[int, ...]] = {}
    exp_in: tuple[tuple[int, ...], ...] | None = None
    exp_out: tuple[tuple[int, ...], ...] | None = None
    for raw_key, raw_val in shapes_field.items():
        key = str(raw_key).strip().lower()
        if key in {"inputs", "outputs"} and isinstance(raw_val, (list, tuple)):
            parsed_seq = [_as_shape_tuple(item) for item in raw_val]
            if parsed_seq and all(item is not None for item in parsed_seq):
                seq = tuple(item for item in parsed_seq if item is not None)
                if key == "inputs":
                    exp_in = seq
                else:
                    exp_out = seq
                continue
        if (shape := _as_shape_tuple(raw_val)) is None:
            continue
        if key in _COARSE_SHAPE_KEYS:
            coarse[key] = shape
        elif key in _OUTPUT_SHAPE_KEYS:
            out_roles[key] = shape
        else:
            in_roles[key] = shape
    return coarse, in_roles, out_roles, exp_in, exp_out


def _role_order_established(
    roles: Mapping[str, Any], order_map: Mapping[str, int]
) -> list[str] | None:
    if any(k not in order_map for k in roles):
        return None
    ranks = [order_map[k] for k in roles]
    if len(set(ranks)) != len(ranks):
        return None
    return sorted(roles, key=lambda k: order_map[k])


def _align_ordered_specs(
    obs_pairs: tuple[tuple[tuple[int, ...], str], ...],
    ordered_specs: tuple[tuple[tuple[int, ...], str | None, bool], ...],
    *,
    ignore_dtype: bool = False,
) -> int:
    """Return 2 on exact ordered match, 1 on shape-matched untyped ambiguity, 0 on mismatch."""
    if len(obs_pairs) != len(ordered_specs):
        return 0
    verdict = 2
    for (obs_shape, obs_dt), (shape, exp_dt, is_tail) in zip(obs_pairs, ordered_specs):
        shape_ok = (
            len(obs_shape) == len(shape) + 1 and obs_shape[1:] == shape
            if is_tail
            else obs_shape == shape
        )
        if not shape_ok:
            return 0
        if ignore_dtype or (exp_dt is not None and obs_dt == exp_dt):
            continue
        if exp_dt is None:
            verdict = 1
        else:
            return 0
    return verdict


def _match_input_dtypes(
    *,
    typed: Mapping[str, Any],
    attrs: Mapping[str, Any],
    input_roles: Mapping[str, tuple[int, ...]],
    explicit_inputs: tuple[tuple[int, ...], ...] | None,
    in_shapes: tuple[tuple[int, ...], ...],
    in_dtypes: tuple[str, ...],
    declared_dtype: str | None,
    primary_in_dt: str | None,
    is_attention_abi: bool,
    is_quantize_family: bool = False,
) -> bool | None:
    explicit_in_dtypes = typed.get("input_dtypes")
    if isinstance(explicit_in_dtypes, (list, tuple)):
        normed_in = [_norm_dtype(raw) for raw in explicit_in_dtypes]
        if not normed_in or any(dt is None for dt in normed_in):
            return None
        return in_dtypes == tuple(dt for dt in normed_in if dt is not None)

    has_complete_roles = bool(input_roles) and (
        explicit_inputs is None or len(explicit_inputs) <= len(input_roles)
    )
    if not has_complete_roles:
        allowed_pos = {primary_in_dt} if primary_in_dt is not None else set()
        has_distinct_role_dt = False
        for attr_key, _ in _DECLARED_INPUT_DTYPE_ROLES:
            if is_quantize_family and attr_key in {"activation_scale_format", "scale_dtype"}:
                continue
            if (decl_dt := _lookup_dtype(typed, attrs, attr_key)) is None:
                continue
            allowed_pos.add(decl_dt)
            if decl_dt not in in_dtypes:
                return False
            if decl_dt != primary_in_dt:
                has_distinct_role_dt = True
        if primary_in_dt is not None:
            checked_in = (
                tuple(
                    dt
                    for s, dt in zip(in_shapes, in_dtypes)
                    if dt not in (_ATTENTION_AUX_DTYPES - allowed_pos) and s != ()
                )
                if is_attention_abi
                else in_dtypes
            )
            if any(dt not in allowed_pos for dt in checked_in):
                return False
            if has_distinct_role_dt:
                return None
            if in_dtypes[0] != primary_in_dt or any(dt != primary_in_dt for dt in checked_in):
                return False
        elif has_distinct_role_dt:
            return None
        return True

    for attr_key, target_roles in _DECLARED_INPUT_DTYPE_ROLES:
        if is_quantize_family and attr_key in {"activation_scale_format", "scale_dtype"}:
            continue
        if (decl_dt := _lookup_dtype(typed, attrs, attr_key)) is None:
            continue
        if not (target_roles & input_roles.keys()):
            if decl_dt not in in_dtypes:
                return False
            return None

    role_dtypes: dict[str, str | None] = {}
    for k in input_roles:
        spec_dt = _lookup_dtype(typed, attrs, *_ROLE_DTYPE_KEYS.get(k, (f"{k}_dtype",)))
        role_dtypes[k] = (
            spec_dt if spec_dt is not None else (None if k in _AUX_ONLY_ROLES else declared_dtype)
        )
    if primary_in_dt is None and all(v is None for v in role_dtypes.values()):
        return True
    for pkey in _PRIMARY_INPUT_KEYS:
        if pkey in input_roles and (p_dt := role_dtypes.get(pkey)) is not None:
            if in_dtypes[0] != p_dt:
                return False
            break
    declared_dt_set = {dt for dt in role_dtypes.values() if dt is not None}
    untyped_aux_shapes = {
        shape
        for k, shape in input_roles.items()
        if role_dtypes[k] is None and k in _AUX_ONLY_ROLES
    }
    aux_skip = _ATTENTION_AUX_DTYPES - declared_dt_set
    obs_seq = tuple(
        (s, dt)
        for s, dt in zip(in_shapes, in_dtypes)
        if not (is_attention_abi and s not in untyped_aux_shapes and (dt in aux_skip or s == ()))
    )
    if input_roles.keys() & _PARTIAL_CACHE_ROLES:
        allowed_dt = {dt for dt in role_dtypes.values() if dt is not None}
        if primary_in_dt is not None:
            allowed_dt.add(primary_in_dt)
        if allowed_dt and any(dt not in allowed_dt for _, dt in obs_seq):
            return False
        return None
    ordered_keys = _role_order_established(input_roles, _INPUT_ROLE_ORDER)
    if ordered_keys is not None:
        ordered_specs = tuple(
            (input_roles[k], role_dtypes[k], k in _TAIL_INPUT_ROLES) for k in ordered_keys
        )
        score = _align_ordered_specs(obs_seq, ordered_specs, ignore_dtype=False)
        return True if score == 2 else (None if score == 1 else False)

    obs_pairs = Counter(obs_seq)
    typed_pairs = Counter(
        (shape, role_dtypes[k]) for k, shape in input_roles.items() if role_dtypes[k] is not None
    )
    if not _counter_leq(typed_pairs, obs_pairs):
        return False
    if any(role_dtypes[k] is None for k in input_roles):
        return None
    if obs_pairs != typed_pairs:
        return False
    return True if len(set(obs_seq)) <= 1 else None

