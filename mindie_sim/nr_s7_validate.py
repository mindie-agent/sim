"""Shared strict pre-coercion validators and BPE envelope bounds.

[Role]
  One place for the "reject before casting" contract used by every hybrid-state
  adapter: booleans, strings, fractional floats, NaN/Inf, negative dimensions
  and non-positive throughput rates are typed ``ModelingError`` rejections.
  Also provides the dependency-free upper bound on byte-level BPE tokens that
  the MCP envelope uses.

[Invariants]
  * ``strict_int`` accepts only ``type(value) is int``; ``True`` and ``2.0``
    and counts beyond finite binary64 range reject rather than coerce.
  * ``strict_number`` never accepts NaN/Inf and never silently coerces a
    string or bool.
  * ``positive_rate`` requires a finite value strictly greater than zero, so
    no later division can raise ``ZeroDivisionError``.
  * ``bpe_token_upper_bound`` is conservative for every byte-level BPE: each
    token covers at least one UTF-8 byte, hence tokens <= UTF-8 byte count.
  * ``sanitize_symbol`` emits a bounded lowercase symbol for arbitrary text:
    no path separator, home marker, quote or control byte survives.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping, Sequence, Set as AbstractSet
from contextvars import ContextVar
from functools import lru_cache, wraps
import inspect
import math
import re
import sys
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from mindie_sim.schema.common import ModelingError

PARALLEL_MINIMUM = 1; PARALLEL_ZERO_MINIMUM = 0; _NEG_ONE = -1; _ZERO_F = 0.0; _SYM_LIMIT = 24; _CODE_LIMIT = 40
_SAFE_SYMBOL = re.compile(r"[^a-z0-9_.:-]+")
_DTYPE_BYTES = {"fp32": 4.0, "float32": 4.0, "fp16": 2.0, "float16": 2.0,
                "bf16": 2.0, "bfloat16": 2.0, "int8": 1.0, "fp8": 1.0,
                "e4m3": 1.0, "e5m2": 1.0, "int32": 4.0, "int64": 8.0}
# All modeled counts eventually enter binary64 byte/time arithmetic. Reject
# unrepresentable integers before allocation, iteration or a disabled branch.
_MAX_NUMBER = (1 << 1024) - (1 << 971)
_RECORD_SCOPE = ContextVar("nr_s7_record_validation", default=None)
_MISSING = object()
_POSITIVE_RECORD_FIELDS = {
    "ModelSpec": {"layers", "hidden_size", "num_heads", "head_dim", "vocab_size"},
    "MlaSpec": {"kv_lora_rank", "qk_nope_head_dim", "v_head_dim"},
    "DraftSpec": {"layers", "hidden_size", "intermediate_size", "num_heads",
                  "num_kv_heads", "head_dim", "q_width", "kv_width", "block_size"}}

# Counts/bytes not listed here may be zero (dense MoE, absent buffers, latency,
# empty work). These records lacked scalar domains in their own constructors.
_SCALAR_RECORD_DOMAINS = {
    "DecodeCostModel": {"hbm_bw", "cube_peak", "vector_peak", "layers", "hidden",
        "element_bytes", "context_tokens", "tp", "pp", "dp", "pcp", "dcp",
        "ep", "etp", "expert_dp", "kv_heads", "head_dim", "query_heads"},
    "StepBreakdown": {"batch_tokens", "active_replicas", "pp_microbatches"},
    "GlmKdaExecution": {"key_dim", "value_dim", "conv_kernel", "block_size", "mamba_block_size"},
    "Mte2ReadModel": {"tokens", "logical_heads", "head_bytes", "cacheline_bytes", "pages_used",
        "nd_transactions", "nz_transactions", "nd_dma_ops", "nz_dma_ops",
        "useful_bytes", "nd_fetched_bytes", "nz_fetched_bytes"},
    "KimiK3DsparkGrouping": set(),
    "DsparkSwaFootprint": {"row_bytes_per_token"},
    "MemoryLedger": {"pp", "logits_per_sequence"},
}
_RECORD_ENUMS = {
    "RecoverSsmValidation": {"expected_cache_mode": {"none", "align"}},
    "GlmKdaExecution": {"cache_mode": {"none", "align"}},
    "KimiK3DsparkGrouping": {"mode": {"source_uniform_grouping",
        "analytical_fallback_split", "analytical_noncausal_split"}},
    "StepBreakdown": {"scaling_mode": {"fixed_per_dp_batch", "fixed_global_batch"}},
    "Software": {"graph_capture_mode": {"none", "full", "piecewise"}},
}
_RECORD_MODULES = {
    "schema.model": ("RequestSpec", "WorkloadSpec", "SloSpec", "ModelSpec", "MlaSpec", "LayerPlan"),
    "catalog.types": ("DraftSpec",), "schema.case": ("Software", "Case"),
    "schema.parallel": ("ParallelSpec",), "schema.hardware": ("Device", "Hardware"),
    "memory_ledger_types": ("MemoryComponent", "MemoryBreakdown"),
    "memory_breakdown": ("MemoryLedger",),
    "parallel_decode_cost": ("DecodeCostModel", "StepBreakdown"),
    "recoverssm_config": ("RecoverSsmValidation", "GlmKdaExecution"),
    "kv_nz_read_model": ("Mte2ReadModel",),
    "dspark_kv_physics": ("KimiK3DsparkGrouping", "DsparkSwaFootprint"),
}


@lru_cache(None)
def _record_bases(cls):
    """Inherit every registered base contract, matched by class identity.

    Bases already exist in loaded modules; avoid circular schema imports.
    Names label the rules, but a subclass's name never selects its contract.
    """
    return tuple(base for base in cls.__mro__ if any(
        base is getattr(sys.modules.get(f"mindie_sim.{module}"), name, None)
        for module, names in _RECORD_MODULES.items() for name in names))


@lru_cache(None)
def _record_kinds(cls):
    return tuple(base.__name__ for base in _record_bases(cls))


def _request_int(name, value, *, minimum=PARALLEL_ZERO_MINIMUM, reason="workload"):
    if not _matches_type(value, int):
        raise ModelingError(f"{reason}_{name}_invalid:type={type(value).__name__}")
    return _int_dom(name, value, minimum=minimum, reason=reason)


def _validate_request(value):
    """One request contract for construction and bypass-resistant consumers."""
    for name, annotation, owner_kinds in _record_hints(type(value)):
        item = getattr(value, name, _MISSING)
        valid = (isinstance(item, str) if "RequestSpec" in owner_kinds and name == "request_id"
                 else _matches_type(item, annotation))
        if item is _MISSING or not valid:
            # Even formatting an enormous int can raise ValueError in Python.
            raise ModelingError(f"workload_{name}_invalid:type={type(item).__name__}")
    if not value.request_id.strip():
        raise ModelingError("workload.request_id_invalid:empty")
    try:
        hash(value.request_id)
    except TypeError as error:
        raise ModelingError("workload.request_id_invalid:unhashable") from error
    for name in ("prompt_tokens", "output_tokens", "prefix_tokens", "priority",
                 "text_prompt_tokens", "media_tokens", "chain_id"):
        _request_int(name, getattr(value, name, None),
                     minimum=1 if name in {"prompt_tokens", "output_tokens"} else 0)
    if type(getattr(value, "media_encoded", None)) is not bool:
        raise ModelingError(f"workload_media_encoded_invalid:{getattr(value, 'media_encoded', None)!r}")
    times = tuple(_num_dom(name, getattr(value, name, None), reason="workload")
                  for name in ("arrival_ms", "gap_ms"))
    if value.prefix_tokens > value.prompt_tokens:
        raise ModelingError("workload.prefix_exceeds_prompt")
    if value.media_tokens > value.prompt_tokens:
        raise ModelingError("workload.media_exceeds_prompt")
    if value.text_prompt_tokens > 0 and value.text_prompt_tokens + value.media_tokens != value.prompt_tokens:
        raise ModelingError("workload.text_media_prompt_mismatch")
    return times


def _validate_workload(value):
    """Validate before hashing IDs; snapshot public sequence inputs immutably."""
    from mindie_sim.schema.model import RequestSpec
    requests = getattr(value, "requests", None)
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes, bytearray)):
        raise ModelingError("workload.requests_invalid")
    requests = tuple(requests)
    if not requests or any(not isinstance(item, RequestSpec) for item in requests):
        raise ModelingError("workload.requests_invalid")
    for request in requests:
        _validate_request(request)
    if not isinstance(getattr(value, "synthetic", None), bool):
        raise ModelingError("workload.synthetic_invalid")
    ids = [request.request_id for request in requests]
    try:
        unique = set(ids)
    except TypeError as error:
        raise ModelingError("workload.request_id_invalid:unhashable") from error
    if len(ids) != len(unique):
        raise ModelingError("workload.request_id_duplicate")
    metadata = getattr(value, "metadata", None)
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes, bytearray)):
        raise ModelingError("workload.metadata_invalid")
    pairs = []
    for pair in metadata:
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray))
                or len(pair) != 2 or any(not isinstance(item, str) for item in pair)):
            raise ModelingError("workload.metadata_invalid")
        pairs.append(tuple(pair))
    return requests, tuple(pairs)


def _validate_record_domains(value):
    """Apply all inherited record domains, including accepted subclasses."""
    for base in _record_bases(type(value)):
        _validate_record_domain(value, base)


def _validate_record_domain(value, base):
    """Shared constructor/nested-record domains (zero optional counts remain legal)."""
    kind = base.__name__
    if kind == "RequestSpec":
        _validate_request(value)
    elif kind == "WorkloadSpec":
        _validate_workload(value)
    elif kind == "SloSpec":
        for field in fields(base):
            item = getattr(value, field.name)
            if item is not None:
                _pos_dom(f"SloSpec.{field.name}", item)
    elif kind == "ModelSpec":
        _pos_dom("ModelSpec.norm_eps", value.norm_eps)
        _num_dom("ModelSpec.state_bytes_per_request", value.state_bytes_per_request)
        for fraction in value.expert_routing:
            _num_dom("ModelSpec.expert_routing", fraction)
    elif kind == "LayerPlan":
        _num_dom("LayerPlan.state_bytes", value.state_bytes)
    elif kind == "Software":
        _pos_dom("Software.mtp_acceptance_length", value.mtp_acceptance_length)
        for tokens, accepted in value.mtp_acceptance_by_output_tokens:
            _int_dom("Software.mtp_output_tokens", tokens, minimum=PARALLEL_MINIMUM)
            _pos_dom("Software.mtp_accepted_tokens", accepted)
    if kind in _SCALAR_RECORD_DOMAINS:
        for field in fields(base):
            item = getattr(value, field.name)
            if type(item) not in (int, float) or field.name == "kda_gate_lower_bound":
                continue
            label = f"{kind}.{field.name}"
            if field.name in _SCALAR_RECORD_DOMAINS[kind]:
                _pos_dom(label, item)
            elif kind == "StepBreakdown" and field.name == "t_non_moe_ms" and value.pp_microbatches > PARALLEL_MINIMUM:
                _num_dom(label, item, minimum=-math.inf)
            else:
                _num_dom(label, item)
        if kind == "DecodeCostModel":
            if value.moe_layers > value.layers or value.top_k > value.num_experts:
                raise ModelingError("nr_s7_DecodeCostModel_moe_counts_invalid")
            if value.moe_layers and (value.top_k < 1 or value.num_experts < 1):
                raise ModelingError("nr_s7_DecodeCostModel_moe_dimensions_invalid")
            for _, offsets in value.axis_endpoint_offsets:
                for offset in offsets:
                    _int_dom("axis_endpoint_offset", offset)
        if kind == "GlmKdaExecution":
            strict_dtype("GlmKdaExecution.activation_dtype", value.activation_dtype)
            strict_text("GlmKdaExecution.kind", value.kind)
        if kind == "StepBreakdown":
            if value.straggler_ratio < 1 or any(not 0 <= getattr(value, name) <= 1
                    for name in ("active_expert_fraction", "host_bound_ratio")):
                raise ModelingError("nr_s7_StepBreakdown_fraction_invalid")
            from mindie_sim.parallel_schedule import validate_overlap_mode
            validate_overlap_mode(value.overlap_mode)
            for time in value.pp_stage_ms:
                _num_dom("step_schedule_ms", time)
            for s_name, time in value.schedule:
                s_min = -math.inf if s_name == "non_moe" and value.pp_microbatches > 1 else 0.0
                _num_dom("step_schedule_ms", time, minimum=s_min)
        if kind == "MemoryLedger":
            for name in ("sharded", "aux", "transient_base", "workspace", "cache_one",
                         "stage_capacity", "comm_rank", "graph_rank", "capacity_rank"):
                for amount in getattr(value, name):
                    validator = _pos_dom if name in {"stage_capacity", "capacity_rank"} else _num_dom
                    validator(f"MemoryLedger.{name}", amount)
            for row in value.aux_rank:
                for amount in row:
                    _num_dom("MemoryLedger.aux_rank", amount)
    for name, allowed in _RECORD_ENUMS.get(kind, {}).items():
        if getattr(value, name) not in allowed:
            raise ModelingError(f"nr_s7_{kind}.{name}_invalid:kind")


_ATOMIC = (int, float, str, bool, bytes, type(None))


@lru_cache(None)
def _ann_parts(annotation: Any):
    return get_origin(annotation), get_args(annotation)


def _matches_type(value: Any, annotation: Any) -> bool:
    """Check raw types recursively, without casts or equality on untyped data."""
    if annotation is Any or annotation is object:
        return True
    cls = type(value)
    if annotation is int:
        return cls is int and abs(value) <= _MAX_NUMBER
    if annotation is bool or annotation is str:
        return cls is annotation
    if annotation is float:
        return ((cls is int and abs(value) <= _MAX_NUMBER)
                or (cls is float and math.isfinite(value)))
    if type(annotation) is type:
        return cls is annotation or isinstance(value, annotation)
    origin, args = _ann_parts(annotation)
    if origin in (types.UnionType, Union):
        return any(_matches_type(value, item) for item in args)
    if origin is tuple:
        if cls is not tuple and not isinstance(value, tuple):
            return False
        if args and args[_NEG_ONE] is not Ellipsis:
            return len(value) == len(args) and all(
                _matches_type(v, t) for v, t in zip(value, args))
        return not args or all(_matches_type(v, args[PARALLEL_ZERO_MINIMUM]) for v in value)
    if origin is list:
        return (cls is list or isinstance(value, list)) and (
            not args or all(_matches_type(v, args[PARALLEL_ZERO_MINIMUM]) for v in value))
    if origin is Sequence:
        if cls is not tuple and cls is not list and (
                isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence)):
            return False
        return not args or all(_matches_type(v, args[PARALLEL_ZERO_MINIMUM]) for v in value)
    if origin in (dict, Mapping):
        ok = (cls is dict) if origin is dict else (cls is dict or isinstance(value, Mapping))
        if not ok and origin is dict:
            ok = isinstance(value, dict)
        return ok and (not args or all(
            _matches_type(k, args[PARALLEL_ZERO_MINIMUM]) and _matches_type(v, args[PARALLEL_MINIMUM])
            for k, v in value.items()))
    if origin in (set, frozenset, AbstractSet):
        return isinstance(value, origin) and (not args or all(
            _matches_type(item, args[0]) for item in value))
    return isinstance(value, origin or annotation)


@lru_cache(None)
def _record_hints(cls):
    """Keep every registered base field/type, then check subclass additions."""
    required = []
    for base in _record_bases(cls):
        hints = get_type_hints(base)
        required.extend((field.name, hints[field.name], frozenset({base.__name__}))
                        for field in fields(base))
    names = {name for name, _, _ in required}
    extra = [field for field in fields(cls) if field.name not in names]
    if extra:
        hints = get_type_hints(cls)
        required.extend((field.name, hints[field.name], frozenset()) for field in extra)
    if "Case" in _record_kinds(cls):
        required.sort(key=lambda pair: pair[0] != "workload")
    return tuple(required)


def _validate_nested(value, path, seen=None):
    """Validate supplied records, keeping Any/object opaque and one-call memo."""
    if type(value) in _ATOMIC:
        return
    if seen is None:
        seen = _RECORD_SCOPE.get()
    if seen is None:
        seen = {}
    if id(value) in seen:
        return
    if is_dataclass(value) and not isinstance(value, type):
        seen[id(value)] = value
        for name, annotation, owner_kinds in _record_hints(type(value)):
            item = getattr(value, name, _MISSING)
            label = f"{path}.{name}"
            # Ledger detail is an extensible diagnostic mapping in practice;
            # its legacy float annotation does not describe lists/strings.
            if owner_kinds & {"MemoryComponent", "MemoryBreakdown"} and name == "detail":
                annotation = tuple[tuple[str, Any], ...]
            if owner_kinds & {"Software", "ModelSpec"} and name in {"metadata", "catalog_facts"}:
                annotation = Mapping[str, Any] | tuple[tuple[str, Any], ...]
            if "RequestSpec" in owner_kinds and name == "request_id":
                annotation = Any  # The request domain accepts isinstance(str).
            if "Device" in owner_kinds and name in {"rates", "compute_peaks"}:
                if _matches_type(item, tuple[tuple[str, Any], ...]):
                    for key, rate in item:
                        if not _matches_type(rate, float):
                            raise ModelingError(f"nr_s7_{label}.{key}_invalid:type={type(rate).__name__}")
                        # A zero optional link means unequipped; compute and
                        # consumed HBM/Vector supplies are never zero sentinels.
                        validator = _num_dom if name == "rates" and key == "link" else _pos_dom
                        domain = "fp32_flops" if name == "compute_peaks" and key == "fp32" else key
                        validator(f"{label}.{domain}", rate)
                    if not item or (name == "rates" and not {"hbm", "vector"} <= dict(item).keys()):
                        raise ModelingError(f"nr_s7_{label}_invalid:missing")
            if item is _MISSING or not _matches_type(item, annotation):
                raise ModelingError(f"nr_s7_{label}_invalid:type={type(item).__name__}")
            if owner_kinds & {"ModelSpec", "LayerPlan", "MlaSpec", "DraftSpec"}:
                if type(item) is int:
                    minimum = -1 if name.endswith(("_owner_layer_id", "_source_layer_id")) or name == "native_layer_id" else 0
                    if any(name in _POSITIVE_RECORD_FIELDS.get(kind, ()) for kind in owner_kinds):
                        minimum = 1
                    if item < minimum:
                        raise ModelingError(f"nr_s7_{label}_invalid:range")
                if name in {"attention", "attention_kind"} and item not in {"mla", "gqa", "mha", "mqa", "dsa", "swa", "csa", "csa2", "hca", "kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet", "projection_only", "unmodeled"}:
                    raise ModelingError(f"nr_s7_{label}_invalid:kind")
            if "Software" in owner_kinds:
                if type(item) is int and item < 1:
                    raise ModelingError(f"nr_s7_{label}_invalid:range")
                if name in {"pp_layer_partition", "graph_buckets", "graph_capture_keys", "piecewise_capture_keys"} and any(count < 0 for count in item):
                    raise ModelingError(f"nr_s7_{label}_invalid:range")
                if name == "gpu_memory_utilization" and not 0 < item <= 1:
                    raise ModelingError(f"nr_s7_{label}_invalid:range")
            if "ParallelSpec" in owner_kinds and type(item) is int:
                minimum = PARALLEL_ZERO_MINIMUM if name == "redundant_experts_per_rank" else PARALLEL_MINIMUM
                if item < minimum:
                    raise ModelingError(f"nr_s7_{label}_invalid:range")
            if "Hardware" in owner_kinds and name == "dies" and not item:
                raise ModelingError(f"nr_s7_{label}_invalid:empty")
            if "WorkloadSpec" in owner_kinds and name == "requests" and not item:
                raise ModelingError(f"nr_s7_{label}_invalid:empty")
            if "Device" in owner_kinds:
                if name in {"memory_bytes", "compute_units"}:
                    _int_dom(label, item, minimum=PARALLEL_MINIMUM)
                elif name in {"rank", "node", "package", "launch_us"}:
                    _num_dom(label, item)
                elif name in {"buffers", "resource_units", "buffer_units"}:
                    for key, count in item:
                        _int_dom(f"{label}.{key}", count, minimum=PARALLEL_ZERO_MINIMUM if name == "buffers" else PARALLEL_MINIMUM)
            if annotation is not Any and annotation is not object and type(item) not in _ATOMIC:
                _validate_nested(item, label, seen)
        _validate_record_domains(value)
        if "MemoryBreakdown" in _record_kinds(type(value)):
            stages = len(value.stage_peak_bytes)
            if not stages or len(value.components) != 6 or not value.per_rank_bytes:
                raise ModelingError(f"nr_s7_{path}_stage_shape_invalid")
            if any(not 0 <= row[1] < stages for row in value.per_rank_bytes):
                raise ModelingError(f"nr_s7_{path}_rank_stage_invalid")
    # Dataclass containers expose members as well as declared fields.
    cls = type(value)
    if cls is tuple or cls is list or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping))):
        seen[id(value)] = value
        for index, item in enumerate(value):
            if type(item) not in _ATOMIC:
                _validate_nested(item, f"{path}[{index}]", seen)
    elif cls is dict or isinstance(value, Mapping):
        seen[id(value)] = value
        for key, item in value.items():
            if type(key) not in _ATOMIC:
                _validate_nested(key, f"{path}.key", seen)
            if type(item) not in _ATOMIC:
                _validate_nested(item, path, seen)
    elif cls in (set, frozenset) or isinstance(value, AbstractSet):
        seen[id(value)] = value
        for item in value:
            if type(item) not in _ATOMIC:
                _validate_nested(item, path, seen)


def typed_inputs(target=None, *, errors=None):
    """Enforce annotated argument types before entering a public boundary."""
    if target is None:
        return lambda value: typed_inputs(value, errors=errors)
    def wrap(function, owner=None):
        signature = inspect.signature(function)
        params = tuple(signature.parameters.values())
        pos = tuple(p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        req_pos = sum(PARALLEL_MINIMUM for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)
        fast_pos = (all(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in params)
                    and not any(p.kind is p.KEYWORD_ONLY and p.default is p.empty for p in params))
        is_init = function.__name__ == "__init__"
        hints = None

        @wraps(function)
        def checked(*args, **kwargs):
            nonlocal hints
            if hints is None:
                hints = (get_type_hints(owner) if is_init
                         else get_type_hints(function, localns=vars(owner) if owner else None))
            if fast_pos and not kwargs and req_pos <= len(args) <= len(pos):
                items = tuple(zip(pos, args))
            else:
                try:
                    items = tuple(signature.bind(*args, **kwargs).arguments.items())
                except TypeError as error:
                    raise ModelingError(f"nr_s7_{function.__name__}_arguments_invalid:{error}") from error
            for name, value in items:
                if name == "minimum" and hints.get(name) is float and type(value) is float and value == -math.inf:
                    continue  # Explicit unbounded lower limit of strict_number.
                if name in hints and not _matches_type(value, hints[name]):
                    code = (errors or {}).get(name, f"nr_s7_{name}_invalid")
                    raise ModelingError(f"{code}:type={type(value).__name__}")
            token = _RECORD_SCOPE.set({}) if _RECORD_SCOPE.get() is None else None
            try:
                for name, value in items:
                    if type(value) not in _ATOMIC and name in hints and hints[name] not in (Any, object):
                        _validate_nested(value, name)
                result = function(*args, **kwargs)
                if is_init:
                    _validate_record_domains(args[PARALLEL_ZERO_MINIMUM] if args else dict(items)["self"])
                return result
            except OverflowError as error:
                raise ModelingError(f"nr_s7_{function.__name__}_invalid:overflow") from error
            finally:
                if token is not None:
                    _RECORD_SCOPE.reset(token)
        return checked

    if isinstance(target, type):
        for name, member in tuple(vars(target).items()):
            if name == "__init__" or not name.startswith("_"):
                if isinstance(member, classmethod):
                    setattr(target, name, classmethod(wrap(member.__func__, target)))
                elif isinstance(member, property) and member.fget is not None:
                    setattr(target, name, property(wrap(member.fget, target),
                                                   member.fset, member.fdel, member.__doc__))
                elif inspect.isfunction(member):
                    setattr(target, name, wrap(member, target))
        return target
    return wrap(target)


@typed_inputs
def typed_sequence(name: str, value: Any, element_type: type, *,
                   nonempty: bool = False) -> tuple:
    """Validate the container and every element before tuple normalization."""
    if not _matches_type(value, Sequence[element_type]) or (nonempty and not value):
        raise ModelingError(f"nr_s7_{name}_invalid")
    _validate_nested(value, name)
    return tuple(value)


def _int_dom(name: str, value: Any, minimum: int = PARALLEL_ZERO_MINIMUM, reason: str = "nr_s7") -> int:
    if type(value) is not int or value < minimum or abs(value) > _MAX_NUMBER:
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    return value

def _num_dom(name: str, value: Any, minimum: float = _ZERO_F, allow_none: bool = False, reason: str = "nr_s7") -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ModelingError(f"{reason}_{name}_invalid:overflow") from error
    if not math.isfinite(number) or number < minimum:
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    return number

def _pos_dom(name: str, value: Any, reason: str = "nr_s7") -> float:
    number = _num_dom(name, value, _ZERO_F, False, reason)
    if number is None or number <= _ZERO_F:
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    return number

@typed_inputs
def strict_int(name: str, value: Any, *, minimum: int = PARALLEL_ZERO_MINIMUM, reason: str = "nr_s7") -> int:
    """Return an exact non-negative ``int`` or raise before any cast."""
    return _int_dom(name, value, minimum, reason)

@typed_inputs
def strict_bool(name: str, value: Any, *, reason: str = "nr_s7") -> bool:
    if type(value) is not bool:
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    return value

@typed_inputs
def strict_number(name: str, value: Any, *, minimum: float = _ZERO_F,
                  allow_none: bool = False, reason: str = "nr_s7") -> float | None:
    return _num_dom(name, value, minimum, allow_none, reason)

@typed_inputs
def positive_rate(name: str, value: Any, *, reason: str = "nr_s7") -> float:
    """A finite, strictly positive throughput/rate; zero is rejected."""
    return _pos_dom(name, value, reason)


@typed_inputs
def strict_text(name: str, value: Any, *, reason: str = "nr_s7") -> str:
    if type(value) is not str or not value.strip():
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    return value


@typed_inputs
def optional_text(name: str, value: Any, *, reason: str = "nr_s7") -> str | None:
    if value is None:
        return None
    return strict_text(name, value, reason=reason)


@typed_inputs
def strict_dtype(name: str, value: Any, *,
                 reason: str = "nr_s7") -> str:
    text = strict_text(name, value, reason=reason).strip().lower()
    if text not in _DTYPE_BYTES:
        raise ModelingError(f"{reason}_{name}_invalid:{value!r}")
    return text


@typed_inputs
def dtype_elem_bytes(name: str, value: Any, *,
                     reason: str = "nr_s7") -> float:
    return _DTYPE_BYTES[strict_dtype(name, value, reason=reason)]


@typed_inputs
def next_power_of_two(value: int) -> int:
    number = _int_dom("value", value, PARALLEL_ZERO_MINIMUM)
    return 1 << max(0, number - 1).bit_length()


@typed_inputs
def ceil_div(numerator: int, denominator: int) -> int:
    _int_dom("numerator", numerator, PARALLEL_ZERO_MINIMUM)
    _int_dom("denominator", denominator, PARALLEL_MINIMUM)
    return (numerator + denominator - 1) // denominator


@typed_inputs
def bpe_token_upper_bound(text: Any) -> int:
    """Conservative upper bound on byte-level BPE tokens of ``text``."""
    if type(text) is not str:
        raise ModelingError(f"nr_s7_bpe_text_invalid:{type(text).__name__}")
    return len(text.encode("utf-8"))


@typed_inputs
def sanitize_symbol(value: Any, *, limit: int = _SYM_LIMIT) -> str:
    """Map arbitrary text to a bounded ``[a-z0-9_.:-]`` symbol."""
    limit = _int_dom("limit", limit, PARALLEL_MINIMUM)
    if type(value) is not str:
        return "invalid"
    lowered = value.strip().lower()
    if not lowered or any(marker in lowered for marker in ("/", "\\", "~", "$")):
        return "empty" if type(value) is str and not lowered else "symbolic"
    cleaned = _SAFE_SYMBOL.sub("-", lowered).strip("-")
    if not cleaned or any(marker in cleaned for marker in ("..", "/", "\\", "~", "$")):
        return "symbolic"
    return cleaned[:limit]


@typed_inputs
def sanitize_code(value: Any, *, limit: int = _CODE_LIMIT) -> str:
    """Bounded diagnostic-code sanitizer preserving known code characters."""
    limit = _int_dom("limit", limit, PARALLEL_MINIMUM)
    if type(value) is not str or not value:
        return "unknown"
    if len(value) > 80 or any(marker in value for marker in
                              ("/", "\\", "~", "$")) or not value.isascii():
        return "symbolic"
    cleaned = re.sub(r"[^a-z0-9_:,.=-]+", "", value.strip().lower())
    return cleaned[:limit] or "unknown"


@typed_inputs
def stage_of_layer(partition: tuple[int, ...], index: int) -> int:
    """Stage owning a contiguous layer index for a declared PP partition."""
    _int_dom("layer_index", index, PARALLEL_ZERO_MINIMUM)
    if not isinstance(partition, tuple) or not partition:
        raise ModelingError("nr_s7_partition_invalid")
    for count in partition:
        _int_dom("partition_count", count, PARALLEL_ZERO_MINIMUM)
    cursor = 0
    for stage, count in enumerate(partition):
        cursor += count
        if index < cursor:
            return stage
    raise ModelingError(f"nr_s7_layer_index_outside_partition:{index}")


@typed_inputs
@dataclass(frozen=True)
class RestoreExtent:
    """Shared exact restore-extent contract for prefix-cache entry points."""

    prefix_hit_tokens: int
    prompt_tokens: int
    block_size: int
    hash_block_size: int
    block_hash_count: int
    exact: bool = True

    def __post_init__(self) -> None:
        _int_dom("prefix_hit_tokens", self.prefix_hit_tokens, PARALLEL_ZERO_MINIMUM)
        _int_dom("prompt_tokens", self.prompt_tokens, PARALLEL_ZERO_MINIMUM)
        _int_dom("block_size", self.block_size, PARALLEL_MINIMUM)
        _int_dom("hash_block_size", self.hash_block_size, PARALLEL_MINIMUM)
        _int_dom("block_hash_count", self.block_hash_count, PARALLEL_ZERO_MINIMUM)
        if type(self.exact) is not bool:
            raise ModelingError(f"nr_s7_exact_invalid:{self.exact!r}")
        if self.block_size % self.hash_block_size:
            raise ModelingError("nr_s7_restore_block_hash_alignment_invalid")
        if self.prefix_hit_tokens > self.prompt_tokens:
            raise ModelingError(
                "nr_s7_restore_hit_exceeds_prompt:"
                f"{self.prefix_hit_tokens}>{self.prompt_tokens}")
        if self.exact and self.prefix_hit_tokens % self.block_size:
            raise ModelingError(
                "nr_s7_restore_requires_full_blocks:"
                f"{self.prefix_hit_tokens}%{self.block_size}")

    @property
    def blocks(self) -> int:
        return (self.prefix_hit_tokens // self.block_size
                if self.exact else 0)

    @property
    def stored_tokens(self) -> int:
        return self.blocks * self.block_size

    def validated_blocks(self) -> int:
        """Full blocks that are both complete and backed by a block hash."""
        count = self.prefix_hit_tokens // self.block_size
        if count > self.block_hash_count:
            raise ModelingError(
                "nr_s7_restore_missing_block_hashes:"
                f"{count}>{self.block_hash_count}")
        return count


__all__ = ["RestoreExtent", "bpe_token_upper_bound", "ceil_div",
           "dtype_elem_bytes", "next_power_of_two", "optional_text",
           "positive_rate", "sanitize_code", "sanitize_symbol",
           "stage_of_layer", "strict_bool", "strict_dtype", "strict_int",
           "strict_number", "strict_text", "typed_inputs", "typed_sequence"]
