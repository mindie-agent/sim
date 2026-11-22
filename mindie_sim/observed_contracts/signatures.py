"""Observed native signature extraction, matching and tensor contract checks.

[Role] Canonicalize observed kernel signatures, match captured signatures to
  declared operator contracts, and emit tensor-contract diagnostics.
[Invariants] Deterministic signature normalization with zero silent drops.
[Timeouts & Locks] Pure in-memory matching; no I/O or locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections import Counter
from typing import Any as Event
from typing import Mapping

from .vocabulary import (
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


def _match_role_shapes(
    *,
    typed: Mapping[str, Any],
    attrs: Mapping[str, Any],
    coarse_shapes: Mapping[str, tuple[int, ...]],
    input_roles: Mapping[str, tuple[int, ...]],
    output_roles: Mapping[str, tuple[int, ...]],
    explicit_inputs: tuple[tuple[int, ...], ...] | None,
    explicit_outputs: tuple[tuple[int, ...], ...] | None,
    in_shapes: tuple[tuple[int, ...], ...],
    in_dtypes: tuple[str, ...],
    out_shapes: tuple[tuple[int, ...], ...],
    out_dtypes: tuple[str, ...],
    declared_dtype: str | None,
    is_attention_abi: bool,
) -> tuple[bool | None, bool]:
    """Validate input/output shape signatures; return ``(verdict, unresolved_order)``."""
    unresolved_order = False
    has_explicit_roles = bool(
        input_roles or output_roles or explicit_inputs is not None or explicit_outputs is not None
    )
    if not has_explicit_roles:
        if not coarse_shapes:
            return False, False
        coarse_set = set(coarse_shapes.values())
        if any(s not in coarse_set for s in (*in_shapes, *out_shapes)):
            return False, False
        if not (len(in_shapes) == 1 and len(out_shapes) == 1 and in_shapes[0] == out_shapes[0]):
            unresolved_order = True
        return True, unresolved_order

    if explicit_outputs is not None:
        if out_shapes != explicit_outputs:
            return False, False
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
                return False, False
        else:
            if Counter(effective_out) != Counter(output_roles.values()):
                return False, False
            if len(set(effective_out)) > 1:
                unresolved_order = True
    elif out_shapes:
        coarse_set = set(coarse_shapes.values())
        if not coarse_set:
            return None, False
        if any(s not in coarse_set for s in out_shapes):
            return False, False
        if len(out_shapes) > 1:
            unresolved_order = True

    if explicit_inputs is not None:
        if in_shapes != explicit_inputs:
            return False, False
    elif input_roles:
        for pkey in _PRIMARY_INPUT_KEYS:
            if pkey in input_roles and in_shapes:
                if in_shapes[0] != input_roles[pkey]:
                    return False, False
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
                    return False, False
            else:
                obs_counts = Counter(s for s, _ in obs_seq)
                if obs_counts != Counter(input_roles.values()):
                    return False, False
                if len(set(s for s, _ in obs_seq)) > 1:
                    unresolved_order = True
    elif in_shapes:
        coarse_set = set(coarse_shapes.values())
        if not coarse_set:
            return None, False
        if any(s not in coarse_set for s in in_shapes):
            return False, False
        if len(in_shapes) > 1:
            unresolved_order = True
    return True, unresolved_order


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

    shape_verdict, unresolved_order = _match_role_shapes(
        typed=typed, attrs=attrs, coarse_shapes=coarse_shapes,
        input_roles=input_roles, output_roles=output_roles,
        explicit_inputs=explicit_inputs, explicit_outputs=explicit_outputs,
        in_shapes=in_shapes, in_dtypes=in_dtypes, out_shapes=out_shapes,
        out_dtypes=out_dtypes, declared_dtype=declared_dtype,
        is_attention_abi=is_attention_abi,
    )
    if shape_verdict is not True:
        return shape_verdict

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


from collections import Counter
from typing import Any, Mapping

from .vocabulary import (
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


from .vocabulary import DTYPES, native_tensor_contract
from .fusion import (
    FUSED_TYPES, fused_contract_diagnostic, fused_contract_scope, observed_fused_contract,
)
from .attention import (
    ATTENTION_CACHE_TYPES, attention_contract_diagnostic, attention_contract_scope,
    observed_attention_contract,
)
from .fusion import (
    FUSION_TYPES, fusion_contract_diagnostic, fusion_contract_scope, observed_fusion_contract,
)
from .vocabulary import (
    LAYOUT_BODY_TYPES, layout_contract_diagnostic, layout_contract_scope, observed_layout_contract,
)
from .matrix import (
    ROUTER_TYPES, router_contract_diagnostic, router_contract_scope, observed_router_contract,
)
from .matrix import (
    MATRIX_BODY_TYPES, matrix_contract_diagnostic, matrix_contract_scope, observed_matrix_contract,
)

GDN_NATIVE_TYPES = {"ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO"}
MATRIX_TYPES = {"MatMulV2", "MatMulV3"}
SWIGLU_TYPES = {"Swiglu", "SwiGLU", "SwiGlu"}
MOE_INIT_NATIVE_TYPES = {"MoeInitRoutingV3", "aclnnMoeInitRoutingV3"}
NATIVE_BODY_TYPES = (FUSION_TYPES | FUSED_TYPES | ATTENTION_CACHE_TYPES
                     | LAYOUT_BODY_TYPES | ROUTER_TYPES | MATRIX_BODY_TYPES)
SUPPORTED_TYPES = (GDN_NATIVE_TYPES | MATRIX_TYPES | SWIGLU_TYPES
                   | MOE_INIT_NATIVE_TYPES | NATIVE_BODY_TYPES | {
    "CausalConv1d", "DynamicMxQuant", "AddRmsNormDynamicMxQuant",
    "fused_gdn_gating_kernel", "Slice",
    "RmsNorm", "AddRmsNorm", "Relu",
})
_ABSENT = {"", "N/A", "NULL", "DT_UNDEFINED"}

MX_SCALE_BLOCK_ELEMENTS = 64
MX_SCALE_BLOCK_ROUNDING_OFFSET = 63
MOE_INIT_SCALE_GROUP = 32
MOE_INIT_INPUT_SLOTS = 2
MOE_INIT_OUTPUT_SLOTS = 4


def _mx_quant_contract(row):
    native = native_tensor_contract(row)
    if native is None:
        return None
    inputs, outputs = native[2:]
    residual = row.kind == "AddRmsNormDynamicMxQuant"
    if len(inputs) != (3 if residual else 1) or len(outputs) not in (
        {3, 4} if residual else {2}
    ):
        return None
    shape, dtype, layout = inputs[0]
    if (len(shape) != 2 or min(shape) <= 0 or layout != "ND"
            or dtype not in {"bf16", "fp16", "fp32"}):
        return None
    m, k = shape
    scale_index = 2 if residual else 1
    if (outputs[0] != (shape, "fp8_e4m3", "ND")
            or outputs[scale_index] != (
                (m, (k + MX_SCALE_BLOCK_ROUNDING_OFFSET) // MX_SCALE_BLOCK_ELEMENTS, 2),
                "fp8_e8m0", "NCL" if residual else "ND")):
        return None
    if residual:
        if (inputs[1] != inputs[0] or inputs[2] != ((k,), dtype, "ND")
                or outputs[1] != inputs[0]):
            return None
        # The retained fused ABI has an optional, typed-empty rstd slot. It
        # must not hide an unvalidated materialized auxiliary output.
        if len(outputs) == 4 and outputs[3] not in {
            ((), "fp32", "ND"), ((), None, None),
        }:
            return None
    return row.kind, shape, dtype, outputs[0][1], outputs[scale_index][0], "fp8_e8m0"


def _moe_init_routing_contract(row):
    """Verified BF16/INT32-to-E4M3/E8M0 local-expert routing signature."""
    native = native_tensor_contract(row)
    if native is None:
        return None
    inputs, outputs = native[2:]
    if len(inputs) != MOE_INIT_INPUT_SLOTS or len(outputs) != MOE_INIT_OUTPUT_SLOTS:
        return None
    query, ids = inputs
    if (len(query[0]) != 2 or query[1] != "bf16" or query[2] != "ND"
            or len(ids[0]) != 2 or ids[1] != "int32" or ids[2] != "ND"):
        return None
    tokens, width = query[0]
    topk = ids[0][1]
    if (tokens <= 0 or width <= 0 or width % MOE_INIT_SCALE_GROUP
            or ids[0][0] != tokens or topk <= 0):
        return None
    rows = tokens * topk
    experts = outputs[2][0]
    if (outputs[0] != ((rows, width), "fp8_e4m3", "ND")
            or outputs[1] != ((rows,), "int32", "ND")
            or len(experts) != 1 or experts[0] <= 0
            or outputs[2][1:] != ("int64", "ND")
            or outputs[3] != ((rows, width // MOE_INIT_SCALE_GROUP),
                              "fp8_e8m0", "ND")):
        return None
    return (
        "moe_init_routing_mxfp8_active", "MoeInitRoutingV3",
        (query, ids), outputs,
    )


def observed_contract(row):
    if row.kind in MOE_INIT_NATIVE_TYPES:
        return _moe_init_routing_contract(row)
    if row.kind in MATRIX_BODY_TYPES:
        return observed_matrix_contract(row)
    if row.kind in ROUTER_TYPES:
        return observed_router_contract(row)
    if row.kind in LAYOUT_BODY_TYPES:
        return observed_layout_contract(row)
    if row.kind in FUSION_TYPES:
        return observed_fusion_contract(row)
    if row.kind in FUSED_TYPES:
        return observed_fused_contract(row)
    if row.kind in ATTENTION_CACHE_TYPES:
        return observed_attention_contract(row)
    if row.kind in {"DynamicMxQuant", "AddRmsNormDynamicMxQuant"}:
        return _mx_quant_contract(row)
    inputs, outputs = row.input_shapes, row.output_shapes
    if not inputs or not outputs:
        return None
    if len(row.input_dtypes) < len(inputs) or len(row.output_dtypes) < len(outputs):
        return None
    if any(shape and dtype in {"", "N/A", "NULL", "DT_UNDEFINED"}
           for shapes, dtypes in ((inputs, row.input_dtypes), (outputs, row.output_dtypes))
           for shape, dtype in zip(shapes, dtypes)):
        return None
    if row.kind in GDN_NATIVE_TYPES:
        return (row.kind, inputs, outputs, tuple(DTYPES.get(d, d) for d in row.input_dtypes),
                tuple(DTYPES.get(d, d) for d in row.output_dtypes))
    if row.kind == "CausalConv1d":
        if len(inputs) < 7 or len(inputs[0]) != 2 or len(inputs[1]) != 2:
            return None
        if inputs[3] and len(inputs[3]) != 3:
            return None
        shapes = (*inputs[:3], inputs[3][1:] if inputs[3] else (), *inputs[4:7])
        return (row.kind, shapes, outputs, tuple(DTYPES.get(d, d) for d in row.input_dtypes[:7]),
                tuple(DTYPES.get(d, d) for d in row.output_dtypes))
    if row.kind in MATRIX_TYPES:
        if len(inputs) < 2 or len(inputs[0]) != 2 or len(outputs) != 1 or len(outputs[0]) != 2:
            return None
        dtypes = tuple(DTYPES.get(value) for value in (*row.input_dtypes[:2], row.output_dtypes[0]))
        if None in dtypes:
            return None
        m, k = inputs[0]
        om, n = outputs[0]
        if om != m:
            return None
        formats = row.input_formats
        if (len(inputs) > 3 or inputs[1] != (n, k)
                or formats[:2] != ("ND", "ND") or row.output_formats != ("ND",)):
            return None
        if len(inputs) == 3 and inputs[2]:
            bias_dtype = DTYPES.get(row.input_dtypes[2])
            if (inputs[2] != (n,) or formats[2:] != ("ND",)
                    or bias_dtype is None):
                return None
            return ("matrix_bias", (m, n, k), *dtypes, inputs[2], bias_dtype)
        scale = (), ""
        return ("matrix", (m, n, k), *dtypes, *scale)
    if row.kind == "fused_gdn_gating_kernel" and len(inputs) >= 3:
        return ("gdn_gating", inputs[1])
    if (row.kind == "Slice" and len(inputs[0]) in {2, 3}
            and len(outputs) == 1 and len(outputs[0]) == len(inputs[0])):
        # Captured geometry does not include index tensor values or source offsets.
        # Preserve rank so an equivalent-element view cannot become an exact match.
        if any(out <= 0 or out > source for source, out in zip(inputs[0], outputs[0])):
            return None
        dtypes = DTYPES.get(row.input_dtypes[0]), DTYPES.get(row.output_dtypes[0])
        return None if None in dtypes else ("slice_copy", inputs[0], outputs[0], *dtypes)
    if row.kind in {"RmsNorm", "AddRmsNorm"}:
        residual = row.kind == "AddRmsNorm"
        base_inputs = 2 if residual else 1
        if len(inputs) not in {base_inputs, base_inputs + 1} or not inputs[0]:
            return None
        dtype = DTYPES.get(row.input_dtypes[0])
        if dtype is None or outputs[0] != inputs[0] or DTYPES.get(row.output_dtypes[0]) != dtype:
            return None
        if residual and (inputs[1] != inputs[0] or DTYPES.get(row.input_dtypes[1]) != dtype):
            return None
        has_weight = len(inputs) > base_inputs and bool(inputs[-1])
        if has_weight and (inputs[-1] != (inputs[0][-1],) or DTYPES.get(row.input_dtypes[-1]) != dtype):
            return None
        if len(row.input_formats) < len(inputs) or len(row.output_formats) < len(outputs):
            return None
        if any(value != "ND" for value in (*row.input_formats, *row.output_formats)):
            return None
        return ("add_rmsnorm" if residual else "rmsnorm", inputs[0], dtype, has_weight)
    if row.kind == "Relu":
        if len(inputs) != 1 or len(outputs) != 1 or not inputs[0]:
            return None
        dtype = DTYPES.get(row.input_dtypes[0])
        if (dtype is None or outputs[0] != inputs[0]
                or DTYPES.get(row.output_dtypes[0]) != dtype
                or row.input_formats != ("ND",) or row.output_formats != ("ND",)):
            return None
        return ("relu", inputs[0], dtype)
    if row.kind in SWIGLU_TYPES:
        if len(inputs) != 1 or len(outputs) != 1 or not inputs[0] or not outputs[0]:
            return None
        dtype = DTYPES.get(row.input_dtypes[0])
        if (dtype is None or DTYPES.get(row.output_dtypes[0]) != dtype
                or inputs[0][:-1] != outputs[0][:-1] or inputs[0][-1] != 2 * outputs[0][-1]
                or row.input_formats != ("ND",) or row.output_formats != ("ND",)):
            return None
        return ("swiglu", outputs[0], dtype)
    return None


def observed_contract_diagnostic(row):
    """Explain evidence/support gaps without guessing a declaration or occurrence."""
    missing, unparsed = [], []
    event = getattr(row, "event", None)
    for field in ("input_shapes", "output_shapes"):
        if not getattr(row, field):
            raw = str(getattr(event, field, "")).strip().strip('"')
            (missing if raw in _ABSENT else unparsed).append(field)
    if unparsed:
        return {
            "category": "parser_contract_gap", "reason": "unparsed_tensor_shapes",
            "detail": "The source has nonempty tensor descriptors that normalization cannot parse.",
            "fields": unparsed, "operator_type": row.kind,
        }
    if row.kind in MATRIX_BODY_TYPES:
        return matrix_contract_diagnostic(row)
    if row.kind in ROUTER_TYPES:
        return router_contract_diagnostic(row)
    if row.kind in LAYOUT_BODY_TYPES:
        return layout_contract_diagnostic(row)
    if row.kind in FUSION_TYPES:
        return fusion_contract_diagnostic(row)
    if row.kind in FUSED_TYPES:
        return fused_contract_diagnostic(row)
    if row.kind in ATTENTION_CACHE_TYPES:
        return attention_contract_diagnostic(row)
    for direction in ("input", "output"):
        shapes = getattr(row, f"{direction}_shapes") or ()
        dtypes = getattr(row, f"{direction}_dtypes")
        if len(dtypes) < len(shapes) or any(
                shape and dtype in _ABSENT for shape, dtype in zip(shapes, dtypes)):
            missing.append(f"{direction}_dtypes")
        if row.kind in MATRIX_TYPES | SWIGLU_TYPES | {
            "RmsNorm", "AddRmsNorm", "Relu", "DynamicMxQuant", "AddRmsNormDynamicMxQuant",
        }:
            formats = getattr(row, f"{direction}_formats")
            if len(formats) < len(shapes) or any(
                    shape and layout in _ABSENT for shape, layout in zip(shapes, formats)):
                missing.append(f"{direction}_formats")
    if missing:
        return {
            "category": "capture_metadata_missing", "reason": "missing_tensor_metadata",
            "detail": "The captured tensor fields needed by this contract are absent or incomplete.",
            "missing_fields": missing, "operator_type": row.kind,
        }
    if row.kind not in SUPPORTED_TYPES:
        return {
            "category": "contract_support_gap", "reason": "unsupported_operator_contract",
            "detail": "Tensor descriptors are present, but this operator type has no observed contract.",
            "operator_type": row.kind,
        }
    if observed_contract(row) is None:
        return {
            "category": "contract_support_gap", "reason": "unsupported_tensor_contract",
            "detail": ("The operator type is supported, but its captured geometry, dtype or layout "
                       "does not satisfy the implemented tensor contract."),
            "operator_type": row.kind,
        }
    return {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "detail": "Captured tensor descriptors form a contract; this does not identify an occurrence.",
        "operator_type": row.kind,
    }


def observed_contract_scope(row):
    """Expose the comparison boundary independently of successful attribution."""
    if row.kind in MATRIX_BODY_TYPES:
        return matrix_contract_scope(row)
    if row.kind in ROUTER_TYPES:
        return router_contract_scope(row)
    if row.kind in LAYOUT_BODY_TYPES:
        return layout_contract_scope(row)
    if row.kind in FUSION_TYPES:
        return fusion_contract_scope(row)
    if row.kind in FUSED_TYPES:
        return fused_contract_scope(row)
    if row.kind in ATTENTION_CACHE_TYPES:
        return attention_contract_scope(row)
    if row.kind in MOE_INIT_NATIVE_TYPES:
        return {
            "comparison_scope": "active_mxfp8_local_expert_routing",
            "basis": "captured_tensor_descriptors",
            "compared": [
                "bf16_query_shape", "int32_expert_ids_shape",
                "mxfp8_output_shape_dtype", "row_index_shape_dtype",
                "expert_id_shape_dtype", "e8m0_scale_shape_dtype",
            ],
            "not_compared": [
                "expert_id_values", "sort_order", "quant_mode",
                "scalar_parameters", "binary_dispatch", "occurrence_identity",
            ],
        }
    return {
        "comparison_scope": "captured_tensor_contract",
        "basis": "captured_tensor_descriptors",
        "semantic_limits": [
            "Tensor equality alone does not establish layer occurrence or implementation flags.",
        ],
    }
