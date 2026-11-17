"""BIND1 native-call signature parsing and paired shape/dtype matching.

[Role]
  Parse captured semicolon-delimited shape/dtype/format lanes and compare them
  with declared ``typed`` operands while preserving native role slot order and
  per-role dtypes.  ``Event`` is annotation-only (``typing.Any``); this module
  imports no profiler runtime code.

[Invariants]
  * Input/output lanes pair index-by-index; unequal counts fail closed (False)
    and incomplete capture columns resolve to None instead of a guessed match.
  * Every declared operand is required; swapped ABI slots return False, and
    role sets without an established order resolve to None.
  * Contiguous/ND declarations reject non-contiguous Ascend formats after
    case normalization (including ``C1HWNCoC0``).

[Timeouts & Locks]
  Pure in-memory tuple/slot inspection per candidate binding; no I/O or locks.

[Side-Effect Policy]
  Pure functions; does not mutate caller bindings, events, or global state.
"""

from __future__ import annotations

from collections import Counter
from typing import Any as Event
from typing import Mapping

from .signature_matching import (
    _align_ordered_specs,
    _lookup_dtype,
    _match_input_dtypes,
    _parse_declared_shapes,
    _role_order_established,
)
from .signature_vocabulary import (
    _ATTENTION_AUX_DTYPES,
    _AUX_ONLY_ROLES,
    _INPUT_ROLE_ORDER,
    _MISSING_TEXT,
    _NON_CONTIGUOUS_FORMATS,
    _OUTPUT_ROLE_ORDER,
    _PARTIAL_CACHE_ROLES,
    _PRIMARY_INPUT_KEYS,
    _QUANTIZE_FAMILIES,
    _ROLE_DTYPE_KEYS,
    _TAIL_INPUT_ROLES,
    _UNDEFINED_DTYPES,
    _norm_dtype,
)


def parse_paired_lanes(
    shapes_text: str | None,
    dtypes_text: str | None,
    formats_text: str | None = None,
) -> tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...], tuple[str, ...]]:
    """Parse semicolon-separated shape/dtype/format strings into aligned lanes."""
    s_whole = str(shapes_text or "").strip().strip('"')
    d_whole = str(dtypes_text or "").strip().strip('"')
    f_whole = str(formats_text or "").strip().strip('"')
    if s_whole.upper() in _MISSING_TEXT or d_whole.upper() in _MISSING_TEXT:
        return "missing", (), (), ()
    s_parts = s_whole.split(";")
    d_parts = d_whole.split(";")
    if len(s_parts) != len(d_parts):
        return "mismatch", (), (), ()
    has_formats = f_whole.upper() not in _MISSING_TEXT
    f_parts = f_whole.split(";") if has_formats else None
    if f_parts is not None and len(f_parts) != len(d_parts):
        return "mismatch", (), (), ()
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    formats: list[str] = []
    for idx, raw_d in enumerate(d_parts):
        s_cell = s_parts[idx].strip().strip('"')
        d_cell = raw_d.strip().strip('"')
        f_cell = f_parts[idx].strip().strip('"').upper() if f_parts is not None else ""
        if d_cell.upper() in _UNDEFINED_DTYPES:
            if s_cell.upper() in _MISSING_TEXT:
                continue
            return "missing", (), (), ()
        dt = _norm_dtype(d_cell)
        if dt is None or s_cell.upper() in {"N/A", "NULL", "NONE", "UNDEFINED"}:
            return "missing", (), (), ()
        if s_cell == "":
            dims: tuple[int, ...] = ()
        else:
            try:
                dims = tuple(int(value) for value in s_cell.split(",") if value.strip())
            except ValueError:
                return "missing", (), (), ()
            if not dims:
                return "missing", (), (), ()
        shapes.append(dims)
        dtypes.append(dt)
        if f_parts is not None:
            if f_cell in _MISSING_TEXT:
                return "missing", (), (), ()
            formats.append(f_cell)
    if not shapes:
        return "missing", (), (), ()
    return "ok", tuple(shapes), tuple(dtypes), tuple(formats)


def event_signature(
    event: Event,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]] | None:
    """Parsed shape/dtype lanes of a captured event, or None when incomplete."""
    in_status, inputs, input_dtypes, _ = parse_paired_lanes(
        event.input_shapes, event.input_dtypes, event.input_formats
    )
    out_status, outputs, output_dtypes, _ = parse_paired_lanes(
        event.output_shapes, event.output_dtypes, event.output_formats
    )
    if in_status != "ok" or out_status != "ok":
        return None
    return (*inputs, *outputs), (*input_dtypes, *output_dtypes)


def native_call_signature_match(binding: Mapping[str, Any], event: Event) -> bool | None:
    """True/False across role-matched input/output shapes, dtypes, and formats; None if unresolved."""
    in_status, in_shapes, in_dtypes, in_formats = parse_paired_lanes(
        event.input_shapes, event.input_dtypes, event.input_formats
    )
    out_status, out_shapes, out_dtypes, out_formats = parse_paired_lanes(
        event.output_shapes, event.output_dtypes, event.output_formats
    )
    if in_status == "mismatch" or out_status == "mismatch":
        return False
    if in_status != "ok" or out_status != "ok":
        return None

    typed = binding.get("typed") or {}
    attrs = typed.get("attrs") if isinstance(typed.get("attrs"), Mapping) else {}
    parsed_shapes = _parse_declared_shapes(typed.get("shapes"))
    if parsed_shapes is None:
        return False
    coarse_shapes, input_roles, output_roles, explicit_inputs, explicit_outputs = parsed_shapes
    family_lower = str(binding.get("family") or "").strip().lower()
    is_attention_abi = family_lower == "attention" or "query" in input_roles
    is_quantize_family = family_lower in _QUANTIZE_FAMILIES
    declared_dtype = _norm_dtype(typed.get("dtype"))
    primary_in_dt = (
        _lookup_dtype(typed, attrs, "input_dtype", "activation_dtype", "activation_format", "query_dtype")
        or declared_dtype
    )
    declared_out_dtype = _lookup_dtype(typed, attrs, "output_dtype", "output_format", "y_dtype")

    unresolved_order = False
    has_explicit_roles = bool(
        input_roles or output_roles or explicit_inputs is not None or explicit_outputs is not None
    )
    if not has_explicit_roles:
        if not coarse_shapes:
            return False
        coarse_set = set(coarse_shapes.values())
        if any(s not in coarse_set for s in (*in_shapes, *out_shapes)):
            return False
        if not (len(in_shapes) == 1 and len(out_shapes) == 1 and in_shapes[0] == out_shapes[0]):
            unresolved_order = True
    else:
        if explicit_outputs is not None:
            if out_shapes != explicit_outputs:
                return False
        elif output_roles:
            effective_out = out_shapes
            if is_attention_abi and len(out_shapes) > len(output_roles):
                if all(
                    out_shapes[i] == () and out_dtypes[i] in {"fp32", "bf16", "fp16"}
                    for i in range(len(output_roles), len(out_shapes))
                ):
                    effective_out = out_shapes[: len(output_roles)]
            ordered_out = _role_order_established(output_roles, _OUTPUT_ROLE_ORDER)
            if ordered_out is not None:
                if effective_out != tuple(output_roles[k] for k in ordered_out):
                    return False
            else:
                if Counter(effective_out) != Counter(output_roles.values()):
                    return False
                if len(set(effective_out)) > 1:
                    unresolved_order = True
        elif out_shapes:
            coarse_set = set(coarse_shapes.values())
            if not coarse_set:
                return None
            if any(s not in coarse_set for s in out_shapes):
                return False
            if len(out_shapes) > 1:
                unresolved_order = True

        if explicit_inputs is not None:
            if in_shapes != explicit_inputs:
                return False
        elif input_roles:
            for pkey in _PRIMARY_INPUT_KEYS:
                if pkey in input_roles and in_shapes:
                    if in_shapes[0] != input_roles[pkey]:
                        return False
                    break
            role_dt_vals = {
                dt
                for k in input_roles
                if (dt := _lookup_dtype(typed, attrs, *_ROLE_DTYPE_KEYS.get(k, (f"{k}_dtype",))))
                is not None
            }
            if declared_dtype is not None:
                role_dt_vals.add(declared_dtype)
            untyped_aux_shapes = {
                shape
                for k, shape in input_roles.items()
                if k in _AUX_ONLY_ROLES
                and _lookup_dtype(typed, attrs, *_ROLE_DTYPE_KEYS.get(k, (f"{k}_dtype",))) is None
            }
            aux_skip = _ATTENTION_AUX_DTYPES - role_dt_vals
            obs_seq = tuple(
                (s, dt)
                for s, dt in zip(in_shapes, in_dtypes)
                if not (
                    is_attention_abi
                    and s not in untyped_aux_shapes
                    and (dt in aux_skip or s == ())
                )
            )
            if input_roles.keys() & _PARTIAL_CACHE_ROLES:
                unresolved_order = True
            else:
                ordered_in = _role_order_established(input_roles, _INPUT_ROLE_ORDER)
                if ordered_in is not None:
                    shape_specs = tuple(
                        (input_roles[k], None, k in _TAIL_INPUT_ROLES) for k in ordered_in
                    )
                    if _align_ordered_specs(obs_seq, shape_specs, ignore_dtype=True) == 0:
                        return False
                else:
                    obs_counts = Counter(s for s, _ in obs_seq)
                    if obs_counts != Counter(input_roles.values()):
                        return False
                    if len(set(s for s, _ in obs_seq)) > 1:
                        unresolved_order = True
        elif in_shapes:
            coarse_set = set(coarse_shapes.values())
            if not coarse_set:
                return None
            if any(s not in coarse_set for s in in_shapes):
                return False
            if len(in_shapes) > 1:
                unresolved_order = True

    in_dt_verdict = _match_input_dtypes(
        typed=typed,
        attrs=attrs,
        input_roles=input_roles,
        explicit_inputs=explicit_inputs,
        in_shapes=in_shapes,
        in_dtypes=in_dtypes,
        declared_dtype=declared_dtype,
        primary_in_dt=primary_in_dt,
        is_attention_abi=is_attention_abi,
        is_quantize_family=is_quantize_family,
    )
    if in_dt_verdict is False:
        return False
    if in_dt_verdict is None:
        unresolved_order = True

    explicit_out_dtypes = typed.get("output_dtypes")
    if isinstance(explicit_out_dtypes, (list, tuple)):
        normed_out = [_norm_dtype(raw) for raw in explicit_out_dtypes]
        if not normed_out or any(dt is None for dt in normed_out):
            return None
        if out_dtypes != tuple(dt for dt in normed_out if dt is not None):
            return False
    else:
        target_out = declared_out_dtype or declared_dtype
        if target_out is not None:
            checked_out = (
                tuple(dt for s, dt in zip(out_shapes, out_dtypes) if s != ())
                if is_attention_abi
                else out_dtypes
            )
            if out_dtypes[0] != target_out or any(dt != target_out for dt in checked_out):
                return False

    if isinstance(typed.get("input_formats"), (list, tuple)):
        if not in_formats:
            return None
        if in_formats != tuple(str(x).strip().upper() for x in typed["input_formats"]):
            return False
    if isinstance(typed.get("output_formats"), (list, tuple)):
        if not out_formats:
            return None
        if out_formats != tuple(str(x).strip().upper() for x in typed["output_formats"]):
            return False
    layout = str(typed.get("layout") or "").strip().lower()
    if layout:
        if not in_formats or not out_formats:
            return None
        if layout in {"contiguous", "nd"}:
            if any(fmt in _NON_CONTIGUOUS_FORMATS for fmt in (*in_formats, *out_formats)):
                return False
        elif "fractal_nz" in layout and "FRACTAL_NZ" not in in_formats:
            return False
    return None if unresolved_order else True


__all__ = ["event_signature", "native_call_signature_match", "parse_paired_lanes"]
