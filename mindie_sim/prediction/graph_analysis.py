"""Execution graph lowering, caching, verification and cost analysis.

[Role] Parse analysis inputs, cache/project execution graphs, verify physical
  work conservation, and evaluate graph costs on the canonical Engine.
[Invariants] Exact work/byte conservation and deterministic graph caching.
[Timeouts & Locks] Honors ReplayOptions.deadline_mono; no locks.
[Side-Effect Policy] Pure simulation."""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import math
import re

from mindie_sim.schema.workload import RequestView

NO_PROPOSALS = 0
DECODE_BASE_ROWS = 1
FIRST_PARTICIPANT = 0
UNIFORM_COUNT_CARDINALITY = 1


def analysis_view(case, request_id, tokens, context, scope, phase, *,
                  index=0, samples=None, **fields):
    """One analysis view whose target rows match the declared speculative width.

    The frontend performs the single ``q_len + draft_proposal_tokens``
    expansion, so the stored view keeps one base row and the configured
    proposal count.  A probe that does not match the declared verification
    width remains an ordinary non-speculative row count.
    """
    proposal = NO_PROPOSALS
    advanced = NO_PROPOSALS
    accepted = NO_PROPOSALS
    if scope == "model" and phase == "decode" and tokens > DECODE_BASE_ROWS:
        from .efficiency import _resolve_speculative_tokens
        configured, policy_mean = _resolve_speculative_tokens(
            case, output_budget=DECODE_BASE_ROWS, tokens_per_seq=None,
            emitted_tokens_per_seq=None)
        if configured > DECODE_BASE_ROWS and tokens == configured:
            proposal = configured - DECODE_BASE_ROWS
            mean = min(float(tokens),
                       max(float(DECODE_BASE_ROWS), float(policy_mean)))
            advanced = (math.floor((index + DECODE_BASE_ROWS) * mean)
                        - math.floor(index * mean))
            accepted = advanced - DECODE_BASE_ROWS
    return RequestView(
        request_id, tokens - proposal, context,
        (scope == "model") if samples is None else bool(samples), phase,
        draft_proposal_tokens=fields.pop("draft_proposal_tokens", proposal),
        draft_advanced_tokens=fields.pop("draft_advanced_tokens", advanced),
        draft_accepted_tokens=fields.pop("draft_accepted_tokens", accepted),
        **fields)


_REBIND_ATOMIC_TYPES = (int, float, bool, type(None))


def _longest_common_prefix(keys: Sequence[str]) -> str:
    prefix = keys[0]
    for key in keys[1:]:
        while prefix and not key.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def _scan_prefix_presence(root: Any, prefix: str, dc_fields: dict[type, tuple[str, ...]]) -> dict[int, bool]:
    """Memoize which subtrees contain ``prefix`` so untouched branches skip traversal."""
    has_pfx: dict[int, bool] = {}

    def scan(item: Any) -> bool:
        t = type(item)
        if t in _REBIND_ATOMIC_TYPES:
            return False
        if t is str:
            return prefix in item
        oid = id(item)
        cached = has_pfx.get(oid)
        if cached is not None:
            return cached
        found = False
        if t is tuple or t is list:
            for part in item:
                pt = type(part)
                if pt is str:
                    if prefix in part:
                        found = True
                elif pt not in _REBIND_ATOMIC_TYPES and scan(part):
                    found = True
        elif t is dict:
            for k, v in item.items():
                kt, vt = type(k), type(v)
                if (kt is str and prefix in k) or (vt is str and prefix in v):
                    found = True
                if kt not in _REBIND_ATOMIC_TYPES and kt is not str and scan(k):
                    found = True
                if vt not in _REBIND_ATOMIC_TYPES and vt is not str and scan(v):
                    found = True
        elif is_dataclass(item) and not isinstance(item, type):
            fnames = dc_fields.get(t)
            if fnames is None:
                fnames = tuple(f.name for f in fields(t) if f.init)
                dc_fields[t] = fnames
            for name in fnames:
                val = getattr(item, name)
                vt = type(val)
                if vt is str:
                    if prefix in val:
                        found = True
                elif vt not in _REBIND_ATOMIC_TYPES and scan(val):
                    found = True
        elif isinstance(item, str):
            found = prefix in item
        elif isinstance(item, (tuple, list)):
            for part in item:
                if scan(part):
                    found = True
        elif isinstance(item, dict):
            for k, v in item.items():
                if scan(k) or scan(v):
                    found = True
        has_pfx[oid] = found
        return found

    scan(root)
    return has_pfx


def _rebind_with_binding(
    value: Any,
    binding: Mapping[str, str],
    prefix: str,
    pattern: "re.Pattern[str]",
    dc_fields: dict[type, tuple[str, ...]],
    has_pfx: Mapping[int, bool] | None = None,
) -> Any:
    memo: dict[int, Any] = {}

    def _sub(match: "re.Match[str]") -> str:
        return binding[match.group()]

    def visit(item: Any) -> Any:
        t = type(item)
        if t in _REBIND_ATOMIC_TYPES:
            return item
        if t is str:
            if prefix and prefix not in item:
                return item
            return pattern.sub(_sub, item)
        oid = id(item)
        if has_pfx is not None and not has_pfx.get(oid, True):
            return item
        if oid in memo:
            return memo[oid]
        out = transform(item, t)
        memo[oid] = out
        return out

    def transform(item: Any, t: type) -> Any:
        if isinstance(item, str):
            if prefix and prefix not in item:
                return item
            return pattern.sub(_sub, item)
        if t is tuple:
            changed = False
            out_list = []
            for part in item:
                pt = type(part)
                if pt in _REBIND_ATOMIC_TYPES:
                    after = part
                elif pt is str:
                    after = pattern.sub(_sub, part) if (not prefix or prefix in part) else part
                elif has_pfx is not None and not has_pfx.get(id(part), True):
                    after = part
                else:
                    after = visit(part)
                if after is not part:
                    changed = True
                out_list.append(after)
            return tuple(out_list) if changed else item
        if t is list:
            return [visit(part) for part in item]
        if t is dict:
            changed = False
            out_dict = {}
            for k, v in item.items():
                nk = visit(k)
                nv = visit(v)
                if nk is not k or nv is not v:
                    changed = True
                out_dict[nk] = nv
            return out_dict if changed else item
        if is_dataclass(item) and not isinstance(item, type):
            fnames = dc_fields.get(t)
            if fnames is None:
                fnames = tuple(f.name for f in fields(t) if f.init)
                dc_fields[t] = fnames
            updates = {}
            for name in fnames:
                before = getattr(item, name)
                bt = type(before)
                if bt in _REBIND_ATOMIC_TYPES:
                    continue
                if bt is str:
                    if prefix and prefix not in before:
                        continue
                    after = pattern.sub(_sub, before)
                elif has_pfx is not None and not has_pfx.get(id(before), True):
                    continue
                else:
                    after = visit(before)
                if after is not before:
                    updates[name] = after
            return replace(item, **updates) if updates else item
        if isinstance(item, tuple):
            result = tuple(visit(part) for part in item)
            return item if all(a is b for a, b in zip(item, result)) else result
        if isinstance(item, list):
            return [visit(part) for part in item]
        if isinstance(item, dict):
            return {visit(key): visit(part) for key, part in item.items()}
        return item

    return visit(value)


def rebind_analysis_requests(value, source_views, target_views):
    """Rename every complete program record from one replica to another.

    The immutable records carry nested request subsets, draft-row maps, ports
    and typed metadata, so a string rewrite is bounded by the exact declared
    request identities of both replicas.
    """
    binding = {source.request_id: target.request_id
               for source, target in zip(source_views, target_views, strict=True)}
    if not binding:
        return value
    pattern = re.compile(
        r"(?<![\w@])(?:" + "|".join(
            re.escape(key) for key in sorted(binding, key=len, reverse=True))
        + r")(?![\w@])")
    prefix = _longest_common_prefix(tuple(binding))
    return _rebind_with_binding(value, binding, prefix, pattern, {})


def rebind_analysis_requests_many(value, source_views, target_views_by_key):
    """Rebind one canonical replica graph to many target replicas sharing one prefix scan."""
    source_ids = tuple(v.request_id for v in source_views)
    if not source_ids or not target_views_by_key:
        return {k: value for k in target_views_by_key}
    prefix = _longest_common_prefix(source_ids)
    pattern = re.compile(
        r"(?<![\w@])(?:" + "|".join(
            re.escape(key) for key in sorted(set(source_ids), key=len, reverse=True))
        + r")(?![\w@])")
    dc_fields: dict[type, tuple[str, ...]] = {}
    has_pfx = _scan_prefix_presence(value, prefix, dc_fields) if prefix else None
    out = {}
    for replica_key, target_views in target_views_by_key.items():
        binding = {
            src: tgt.request_id
            for src, tgt in zip(source_ids, target_views, strict=True)
        }
        out[replica_key] = _rebind_with_binding(
            value, binding, prefix, pattern, dc_fields, has_pfx=has_pfx,
        )
    return out


from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, Callable

from mindie_sim.schema.common import ModelingError


DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ENTRIES = 128
FREEZE_TAG_LIST = "__list__"
FREEZE_TAG_MAP = "__map__"
FREEZE_TAG_SET = "__set__"
FREEZE_TAG_REPR = "__repr__"


def freeze_value(value: Any) -> Any:
    """Convert nested metadata into a hashable, deeply immutable projection."""
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    if isinstance(value, Mapping):
        return (FREEZE_TAG_MAP, tuple(sorted(
            ((str(key), freeze_value(item)) for key, item in value.items()),
            key=lambda pair: pair[0])))
    if isinstance(value, (list, tuple)):
        return (FREEZE_TAG_LIST, tuple(freeze_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (FREEZE_TAG_SET, tuple(sorted(
            (freeze_value(item) for item in value), key=repr)))
    try:
        hash(value)
        return value
    except TypeError:
        return (FREEZE_TAG_REPR, type(value).__name__, repr(value))


@dataclass(frozen=True)
class GraphKey:
    """Hashable semantic identity of one frontend graph template."""

    scope: str
    signature: tuple
    selection: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "signature": self.signature,
                "selection": self.selection}


@dataclass
class SemanticGraphCache:
    """Bounded caller-owned LRU of immutable semantic graph templates."""

    max_bytes: int = DEFAULT_MAX_BYTES
    max_entries: int = DEFAULT_MAX_ENTRIES
    _entries: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _sizes: dict = field(default_factory=dict, repr=False)
    _bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oversized: int = 0

    def __post_init__(self) -> None:
        if self.max_bytes < 1 or self.max_entries < 1:
            raise ModelingError("semantic_graph_cache_budget_invalid")

    @property
    def total_bytes(self) -> int:
        return int(self._bytes)

    @property
    def entries(self) -> int:
        return len(self._entries)

    def _size(self, value: Any) -> int:
        frozen = freeze_value(value)
        try:
            return len(json.dumps(frozen, default=repr).encode("utf-8"))
        except (TypeError, ValueError):
            return len(repr(frozen).encode("utf-8"))

    def _evict(self) -> None:
        while (self._entries
               and (len(self._entries) > self.max_entries
                    or self._bytes > self.max_bytes)):
            key, _template = self._entries.popitem(last=False)
            self._bytes -= self._sizes.pop(key, 0)
            self.evictions += 1

    def get_or_build(self, key: GraphKey, builder: Callable[[], Any]) -> Any:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        template = builder()
        size = self._size(template)
        if size > self.max_bytes // 2:
            self.oversized += 1
            return template
        self._entries[key] = template
        self._sizes[key] = size
        self._bytes += size
        self._evict()
        return template

    def clear(self) -> None:
        self._entries.clear()
        self._sizes.clear()
        self._bytes = 0
        self.hits = self.misses = self.evictions = self.oversized = 0

    def stats(self) -> dict[str, int]:
        return {"entries": self.entries, "bytes": self.total_bytes,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "oversized": self.oversized}


def semantic_graph_key(
    case: Any, plan: Any, *, scope: str, selection_signature: Any = (),
) -> GraphKey:
    """Build the hardware-independent semantic key of one graph template."""
    runtime = case.runtime
    parallel = runtime.parallel
    model = case.model
    quant = getattr(model, "quant", None)
    signature = freeze_value({
        "schema": "mindie-sim.semantic-graph/1",
        "scope": str(scope),
        "model": {
            "name": model.name, "layers": model.layers,
            "hidden_size": model.hidden_size, "num_heads": model.num_heads,
            "num_kv_heads": model.num_kv_heads, "head_dim": model.head_dim,
            "vocab_size": model.vocab_size, "num_experts": model.num_experts,
            "top_k": model.top_k,
            "quant": getattr(quant, "name", ""),
            "quant_contract": (
                getattr(quant, "weight_format", ""),
                getattr(quant, "activation_format", ""),
                getattr(quant, "expert_weight_format", ""),
                getattr(quant, "kv_dtype", ""),
                getattr(quant, "index_dtype", ""),
                getattr(quant, "engram_table_format", ""),
                tuple(getattr(quant, "exclusions", ()) or ()),
                tuple(getattr(quant, "category_overrides", ()) or ()),
            ) if quant is not None else (),
            "layer_plans": tuple(
                (item.index, item.attention, item.ffn, item.scope,
                 item.cache_kind, item.window, item.compress_ratio,
                 item.num_experts, item.top_k,
                 getattr(item, "residual_block_size", 0),
                 getattr(item, "hc_mult", 1),
                 getattr(item, "hc_lowrank", 0),
                 getattr(item, "hc_sinkhorn_iters", 0),
                 getattr(item, "index_kpool", 1),
                 getattr(item, "kv_owner_layer_id", -1),
                 getattr(item, "index_k_owner_layer_id", -1),
                 getattr(item, "topk_owner_layer_id", -1),
                 getattr(item, "candidate_source_layer_id", -1),
                 getattr(item, "index_source_layer_id", -1),
                 getattr(item, "native_layer_id", None),
                 bool(getattr(item, "engram", False)),
                 bool(getattr(item, "hyper_connections", False)))
                for item in model.layer_plans),
            "auxiliary": freeze_value(getattr(model, "auxiliary", None)),
        },
        "parallel": {
            "dp": parallel.dp, "tp": parallel.tp, "ep": parallel.ep,
            "pp": parallel.pp, "cp": parallel.cp, "pcp": parallel.pcp,
            "o_proj_tp": parallel.o_proj_tp,
            "linear_proj_tp": parallel.linear_proj_tp,
            "lm_head_tp": parallel.lm_head_tp,
            "redundant_experts_per_rank": parallel.redundant_experts_per_rank,
            "cp_plan": parallel.cp_plan,
            "rank_table": parallel.rank_table,
        },
        "features": {
            "collective_startup_mode": runtime.collective_startup_mode,
            "matrix_compute_policy": runtime.matrix_compute_policy,
            "recover_ssm": runtime.recover_ssm,
            "prefix_cache_mode": runtime.prefix_cache_mode,
            "prefix_read_mode": runtime.prefix_read_mode,
            "homogeneous_graph_reuse": runtime.homogeneous_graph_reuse,
            "routing": freeze_value(runtime.routing),
            "metadata": freeze_value(runtime.metadata),
            "attention_policy": freeze_value(runtime.attention_policy),
            "lowering_passes": runtime.lowering_passes,
        },
        "plan": freeze_value(plan),
        "selection": freeze_value(selection_signature),
    })
    return GraphKey(str(scope), signature, freeze_value(selection_signature))


from collections.abc import Mapping


def bounded_event_profile(engine) -> dict:
    """Separate completed physical events from optional debug-trace retention."""
    retained = len(engine.trace)
    return {
        "completed_physical_events": engine.completed_count,
        "completed_tasks": engine.completed_count,
        "trace_retained": retained,
        "trace_dropped": engine.trace_dropped,
        "trace_events": retained + engine.trace_dropped,
    }


def matrix_wrapper_projection(metadata) -> dict:
    """Publish only the selected wrapper's semantic kind/source role.

    The nested metadata is the explicit source-selection result; absent it the
    ordinary matrix row keeps no kind. This is attribution, not a second op.
    """
    wrapper = metadata.get("matrix_wrapper")
    if not isinstance(wrapper, Mapping):
        return {}
    native = wrapper.get("native_kind")
    if not isinstance(native, str) or not native:
        return {}
    result = {"native_kind": native}
    role = wrapper.get("source_role")
    if isinstance(role, str) and role:
        result["source_role"] = role
    source = wrapper.get("source")
    basis = (wrapper.get("source_basis")
             or (source.get("basis") if isinstance(source, Mapping) else None))
    if isinstance(basis, str) and basis:
        result["source_basis"] = basis
    return result


def accel_status(view) -> dict:
    """Stable compact subset of ``Engine.accel_status()`` for graph output."""
    return {
        "accel_backend": view.get("accel_backend"),
        "native_steps": int(view.get("native_steps") or 0),
        "python_steps": int(view.get("python_steps") or 0),
        "fallback_reason": view.get("fallback_reason"),
        "mode": view.get("mode"),
        "enabled": bool(view.get("enabled")),
    }


from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError

SCHEMA = "mindie-sim/collective-participant-profile/1"
_PRIMITIVES = {
    "allgather": "all_gather", "all_gather": "all_gather",
    "reducescatter": "reduce_scatter", "reduce_scatter": "reduce_scatter",
    "alltoall": "all_to_all", "all_to_all": "all_to_all",
    "alltoallv": "all_to_all", "allreduce": "all_reduce",
    "all_reduce": "all_reduce",
}


def _primitive(value: Any) -> str | None:
    token = str(value or "").strip().lower().replace("-", "_")
    return _PRIMITIVES.get(token)


def _dtype(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    return {"bfp16": "bf16", "bfloat16": "bf16", "float16": "fp16",
            "fp16": "fp16", "float32": "fp32", "fp32": "fp32",
            "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3",
            "fp8_e5m2": "fp8_e5m2"}.get(token, token or None)


def normalise(value: Any) -> dict[str, Any] | None:
    """Validate the optional profile without inventing absent fields."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ModelingError("collective_participant_profile_must_be_mapping")
    doc = value.get("participant_profile", value)
    if not isinstance(doc, Mapping):
        raise ModelingError("collective_participant_profile_must_be_mapping")
    schema = doc.get("schema")
    if schema not in (None, SCHEMA):
        raise ModelingError(f"collective_participant_profile_schema_unknown:{schema}")
    calls = []
    for raw in doc.get("calls") or ():
        if not isinstance(raw, Mapping):
            raise ModelingError("collective_participant_profile_call_must_be_mapping")
        primitive = _primitive(raw.get("primitive") or raw.get("collective"))
        if primitive is None:
            raise ModelingError(
                f"collective_participant_profile_primitive_unknown:{raw.get('primitive')}")
        ordinal = raw.get("ordinal")
        if ordinal is not None:
            try:
                ordinal = int(ordinal)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"collective_participant_profile_ordinal_invalid:{ordinal}") from None
        group_size = raw.get("group_size")
        if group_size is not None:
            try:
                group_size = int(group_size)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"collective_participant_profile_group_invalid:{group_size}") from None
        ranks = []
        for item in raw.get("ranks") or raw.get("participants") or ():
            if not isinstance(item, Mapping):
                raise ModelingError(
                    "collective_participant_profile_rank_must_be_mapping")
            rank_id = item.get("rank_id")
            if rank_id is None:
                raise ModelingError("collective_participant_profile_rank_id_missing")
            try:
                rank_id = int(rank_id)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"collective_participant_profile_rank_id_invalid:{rank_id}") from None
            ranks.append({
                "rank_id": rank_id,
                "role": str(item.get("role") or ""),
            })
        calls.append({
            "ordinal": ordinal,
            "primitive": primitive,
            "native_name": (str(raw["native_name"])
                            if raw.get("native_name") is not None else None),
            "group_size": group_size,
            "dtype": _dtype(raw.get("dtype")),
            "ranks": tuple(ranks),
        })
    return {
        "schema": SCHEMA,
        "source": dict(doc.get("source") or {}),
        "calls": tuple(calls),
    }


def select_call(profile: Mapping[str, Any] | None, *, primitive: Any,
                occurrence: int = 0, group_size: int | None = None,
                dtype: Any = None) -> dict[str, Any]:
    """Select one genuine call by primitive and occurrence, never by time."""
    if profile is None:
        return {"resolved": False, "reason": "participant_profile_absent"}
    family = _primitive(primitive)
    if family is None:
        return {"resolved": False, "reason": f"primitive_unknown:{primitive}"}
    matching = [call for call in profile.get("calls", ())
                if call["primitive"] == family]
    if not matching:
        return {"resolved": False, "reason": f"no_source_call_for_primitive:{family}"}
    if occurrence < 0 or occurrence >= len(matching):
        return {"resolved": False,
                "reason": f"source_call_occurrence_out_of_range:{occurrence}:"
                          f"available={len(matching)}"}
    call = matching[occurrence]
    if (group_size is not None and call.get("group_size") is not None
            and int(call["group_size"]) != int(group_size)):
        return {"resolved": False,
                "reason": f"source_group_size_mismatch:{call['group_size']}!={group_size}"}
    expected = _dtype(dtype)
    if (expected is not None and call.get("dtype") is not None
            and call["dtype"] != expected):
        return {"resolved": False,
                "reason": f"source_dtype_mismatch:{call['dtype']}!={expected}"}
    return {
        "resolved": True,
        "basis": "same_primitive_occurrence",
        "primitive": family,
        "occurrence": occurrence,
        "ordinal": call.get("ordinal"),
        "native_name": call.get("native_name"),
        "group_size": call.get("group_size"),
        "dtype": call.get("dtype"),
        "dtype_basis": ("declared_source" if call.get("dtype") is not None
                        else "underdetermined_source"),
        "ranks": [dict(item) for item in call.get("ranks", ())],
    }


import math

from .program import RecipeBook
from .mechanism import PlannedRecipe


def _check_row(row, rel_tol):
    declared, serviced = row["declared_service"], row["serviced"]
    mismatches = {
        name: {"declared": declared.get(name, 0), "executed": serviced.get(name, 0)}
        for name in declared.keys() | serviced.keys()
        if not math.isclose(declared.get(name, 0), serviced.get(name, 0),
                            rel_tol=rel_tol, abs_tol=1e-12)
    }
    finite = math.isfinite(row["duration_ms"]) and row["duration_ms"] >= 0
    released = not any(row["remaining_buffers"].values()) and row["live_tasks"] == 0
    ready_obeyed = not any(row["service_before_input"].values())
    return {"finite_completion": finite, "buffers_released": released,
            "input_dependency_obeyed": ready_obeyed, "work_mismatches": mismatches}


def verify_operator(case, op, build, *, kernels=None, rel_tol=1e-9, input_ready_ms=0):
    """Check the actual public isolated-cost path, not a second recipe executor."""
    from .operator_analysis import operator_costs
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(build, kernels)})
    rows = operator_costs(case, (op,), recipes=recipes,
                          input_ready_ms=input_ready_ms)["operators"]
    checks = [_check_row(item, rel_tol) for item in rows]
    if kernels is None:
        row = rows[0]
        mismatches = checks[0]["work_mismatches"]
    else:
        def total(field):
            keys = {key for item in rows for key in item[field]}
            return {key: sum(item[field].get(key, 0) * item["occurrences"] for item in rows)
                    for key in keys}
        row = {
            "operator": op.to_dict(), "kernels": rows,
            "duration_ms": sum(item["total_ms"] for item in rows),
            "declared_service": total("declared_service"), "serviced": total("serviced"),
            "remaining_buffers": total("remaining_buffers"),
            "service_before_input": total("service_before_input"),
            "live_tasks": sum(item["live_tasks"] for item in rows),
        }
        # Opposite errors in two kernels must not cancel in the display total.
        mismatches = {item["operator"]["op_id"]: check["work_mismatches"]
                      for item, check in zip(rows, checks) if check["work_mismatches"]}
    finite = all(check["finite_completion"] for check in checks)
    released = all(check["buffers_released"] for check in checks)
    ready_obeyed = all(check["input_dependency_obeyed"] for check in checks)
    return {"status": "passed" if finite and released and ready_obeyed and not mismatches else "failed",
            "scope": "resource_contract_only", "finite_completion": finite,
            "buffers_released": released,
            "input_dependency_obeyed": ready_obeyed,
            "kernel_checks": [{"operator_id": item["operator"]["op_id"], **check}
                              for item, check in zip(rows, checks)],
            "work_mismatches": mismatches, "result": row}


from collections import deque
from collections.abc import Mapping
import math
from mindie_sim.coverage import model_coverage

from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from .efficiency.cost import CostModel
from .execution.cohort_traffic import bind_cohort_traffic
from .execution.dispatch import cohorts
from .execution import Execution
from .native_binding import bind_native_call
from .program import RecipeBook
from .structure import parse_case
from .efficiency.exposed_breakdown import (
    CLOCK_GAP_ABS_FLOOR_MS, CLOCK_GAP_LIMIT_FRACTION, ExposedClocks,
    advance_clocks, communication_clocks, control_clocks, cube_clocks, finish_breakdown,
    memory_clocks, unavailable_breakdown, vector_clocks)
from .efficiency.resource_attribution import realized_clock_dag

#: A multi-layer graph that amortizes eager callback-free native adoption.
EAGER_GRAPH_MIN_OPS = 16

RECONCILE_FRACTION = 1e-3
RECONCILE_MIN_MS = 1.0
ROLE_CUBE = "cube"
ROLE_VECTOR = "vector"
ROLE_MEMORY = "memory"
ROLE_COMMUNICATION = "communication"
ROLE_CONTROL = "control"


def _dominant_role(op, *, stage_rates: Mapping[str, Any] | None = None) -> str:
    """Dominant physical service role of one executed occurrence by roofline argmax."""
    link = float(getattr(op, "link_bytes", 0) or 0)
    flops = float(getattr(op, "flops", 0) or 0)
    vec = float(getattr(op, "vector_ops", 0) or 0)
    metadata = dict(getattr(op, "metadata", ()) or ())
    hbm = float(getattr(op, "hbm_bytes", 0) or 0)
    if hbm <= 0.0:
        hbm = sum(float(getattr(op, field, 0) or 0) for field in ("mte2_bytes", "mte3_bytes"))
        hbm += float(metadata.get("mte1_bytes", 0) or 0)
        hbm += float(metadata.get("fixpipe_bytes", 0) or 0)
    if stage_rates:
        peaks = stage_rates.get("compute_peaks") or {}
        dtype = str(getattr(op, "dtype", "") or "bf16").strip().lower()
        cube_rate = float(peaks.get(dtype) or peaks.get("bf16") or peaks.get("fp16") or 0.0)
        vec_rate = float(stage_rates.get("vector") or 0.0)
        hbm_rate = float(stage_rates.get("hbm") or 0.0)
        link_rate = float(stage_rates.get("link") or 0.0)
        times = (
            (ROLE_COMMUNICATION, (link / link_rate) if (link > 0.0 and link_rate > 0.0) else (float("inf") if link > 0.0 else 0.0)),
            (ROLE_CUBE, (flops / cube_rate) if (flops > 0.0 and cube_rate > 0.0) else 0.0),
            (ROLE_VECTOR, (vec / vec_rate) if (vec > 0.0 and vec_rate > 0.0) else 0.0),
            (ROLE_MEMORY, (hbm / hbm_rate) if (hbm > 0.0 and hbm_rate > 0.0) else 0.0),
        )
        best_role, best_time = max(times, key=lambda pair: pair[1])
        if best_time > 0.0:
            return best_role
    if link > 0:
        return ROLE_COMMUNICATION
    if flops > 0:
        return ROLE_CUBE
    if vec > 0:
        return ROLE_VECTOR
    return ROLE_MEMORY if hbm > 0 else ROLE_CONTROL


def _stage_role_rates(execution) -> dict[tuple[int, int], dict[str, Any]]:
    """Extract per-(replica, stage) physical peak rates for dominant-role classification."""
    out: dict[tuple[int, int], dict[str, Any]] = {}
    link_rates = [
        float(res.rate) * 1000.0
        for name, res in execution.engine.resources.items()
        if str(name).startswith("link") and float(res.rate) > 0.0
    ]
    link_rate = max(link_rates, default=0.0)
    for (replica, stage), profile in getattr(execution, "stage_profiles", {}).items():
        hbm_rate = float(profile.rates.get("hbm", (0.0,))[0]) * 1000.0
        vec_rate = float(profile.rates.get("vector", (0.0,))[0]) * 1000.0
        peaks = {str(k).lower(): float(v) for k, v in dict(profile.compute_peaks).items()}
        out[(int(replica), int(stage))] = {
            "hbm": hbm_rate, "vector": vec_rate,
            "link": link_rate, "compute_peaks": peaks,
        }
    return out


def _exposed_breakdown(records, duration_ms, input_ready_ms, replica,
                       service_clocks_by_id=None, stage_rates_by_key=None):
    """Reduce executed occurrence service clocks onto the device-step boundary.

    Intervals are the actual ``[ready, complete]`` spans of one replica's
    executed operators.  Each occurrence contributes its executed service
    interval to its declared dominant physical stage; no byte/FLOP proportion
    is fabricated and no clock is rescaled to the graph span.  The clocks are
    propagated through the realized causal DAG (data, stream, state and
    collective-rendezvous predecessors).  A missing periodic cross-stage wait
    or an unreconciled boundary makes the report explicitly unavailable.
    """
    intervals = []
    for record in records:
        if record["replica"] != replica:
            continue
        ready = record.get("collective_ready_ms", record["ready"].time)
        start = max(0.0, float(ready) - float(input_ready_ms))
        end = max(start, float(record["complete"].time) - float(input_ready_ms))
        intervals.append((start, end, record))
    if not any(record["op"].family != "barrier" for _start, _end, record in intervals):
        return unavailable_breakdown(
            "exposed_operator_intervals_missing", boundary="device_step")
    first_start = min(start for start, _end, _record in intervals)
    span_end = max(end for _start, end, _record in intervals)
    tolerance = RECONCILE_FRACTION * max(RECONCILE_MIN_MS, float(duration_ms))
    if abs(span_end - float(duration_ms)) > tolerance:
        return unavailable_breakdown(
            "unreconciled_execution_boundary", boundary="device_step")

    def service_role(op):
        if op.family == "barrier":
            return ROLE_CONTROL
        stage_rates = (
            (stage_rates_by_key or {}).get((int(replica), int(getattr(op, "stage", 0) or 0)))
            if stage_rates_by_key else None
        )
        return _dominant_role(op, stage_rates=stage_rates)

    def record_id(record, ordinal):
        return str(record.get("id") or f"record{ordinal}")

    identifiers = [record_id(record, ordinal)
                   for ordinal, (_start, _end, record) in enumerate(intervals)]
    if len(set(identifiers)) != len(identifiers):
        return unavailable_breakdown(
            "exposed_occurrence_identity_duplicate", boundary="device_step")
    has_dependency_identity = all(
        "predecessors" in record for _start, _end, record in intervals)
    dependencies = (
        {identifiers[ordinal]: tuple(record["predecessors"])
         for ordinal, (_start, _end, record) in enumerate(intervals)}
        if has_dependency_identity else _serial_dependencies(intervals, identifiers))
    local, waits = {}, {ROLE_COMMUNICATION: 0.0, ROLE_CONTROL: 0.0}
    for (_start, _end, record), node, (wait, elapsed) in zip(
            intervals, identifiers, _causal_wait_ms(intervals, identifiers, dependencies)):
        role = service_role(record["op"])
        supplied = (service_clocks_by_id or {}).get(node)
        service = (supplied if isinstance(supplied, ExposedClocks)
                   else _ROLE_CLOCKS[role](elapsed))
        # Causal wait between the latest in-DAG predecessor and this occurrence's
        # readiness is real exposed time: peer skew for a collective, dispatch /
        # launch / stream gating otherwise.  It is attributed, never stuffed.
        wait_role = ROLE_COMMUNICATION if role == ROLE_COMMUNICATION else ROLE_CONTROL
        waits[wait_role] += wait
        local[node] = advance_clocks((_ROLE_CLOCKS[wait_role](wait),), service)
    _clocks, terminal = realized_clock_dag(
        local, dependencies, allow_external_predecessors=True)
    expected = float(duration_ms)
    if terminal.all_ms <= 0.0:
        if expected > 0.0:
            return unavailable_breakdown(
                "service_clocks_missing", boundary="device_step")
    gap_limit = max(expected * CLOCK_GAP_LIMIT_FRACTION,
                    CLOCK_GAP_ABS_FLOOR_MS)
    if abs(expected - terminal.all_ms) > gap_limit:
        return unavailable_breakdown(
            "unreconciled_service_clock_boundary", boundary="device_step")
    result = finish_breakdown(
        terminal, duration_ms=expected, boundary="device_step",
        control_ms=waits[ROLE_CONTROL])
    basis = result["exposed_breakdown_basis"]
    basis["replica"] = int(replica)
    basis["operator_intervals"] = sum(
        record["op"].family != "barrier" for _start, _end, record in intervals)
    basis["host_launch_ms"] = float(first_start)
    basis["causal_wait_ms"] = {
        "communication_rendezvous": float(waits[ROLE_COMMUNICATION]),
        "control_dispatch": float(waits[ROLE_CONTROL]),
    }
    basis["method"] = "four_clock_service_attribution_v3"
    basis["service_basis"] = (
        "executed_interval_assigned_to_declared_dominant_physical_class_"
        "plus_attributed_causal_wait")
    basis["declared_edges"] = bool(has_dependency_identity)
    basis["realized_edges"] = sum(
        len(predecessors) for predecessors in dependencies.values())
    return result


_ROLE_CLOCKS = {
    ROLE_COMMUNICATION: communication_clocks, ROLE_CUBE: cube_clocks,
    ROLE_VECTOR: vector_clocks, ROLE_MEMORY: memory_clocks,
    ROLE_CONTROL: control_clocks,
}


def _serial_dependencies(intervals, identifiers):
    """No declared edges: keep observed intervals serial in execution order."""
    ordered = [identifier for _interval, identifier in sorted(
        zip(intervals, identifiers), key=lambda pair: (pair[0][0], pair[0][1]))]
    return {node: (() if index == 0 else (ordered[index - 1],))
            for index, node in enumerate(ordered)}


def _causal_wait_ms(intervals, identifiers, dependencies):
    """Per-occurrence ``(causal wait, executed service)`` split of its span.

    The wait runs from the latest in-DAG predecessor completion (roots: the
    replica's input boundary, i.e. host launch) to readiness; service runs from
    the later of readiness and that floor to completion.  With this split the
    realized ``all`` clock of every node equals its completion, so the terminal
    telescopes exactly to the executed device-step boundary.
    """
    complete = {node: end for (_start, end, _record), node in zip(intervals, identifiers)}
    split = []
    for (start, end, _record), node in zip(intervals, identifiers):
        floor = max((complete[pred] for pred in dependencies.get(node, ()) if pred in complete),
                    default=0.0)
        split.append((max(0.0, start - floor), max(0.0, end - max(start, floor))))
    return split


def _name(replica, identifier):
    return f"r{replica}:{identifier}"


def _validate_graph(ops, external):
    available = set(external)
    for op in ops:
        if op.op_id in available:
            raise ValueError(f"graph_duplicate_node_or_input:{op.op_id}")
        missing = set(op.deps) - available
        if missing:
            raise ValueError(f"graph_dependency_unavailable:{op.op_id}:{','.join(sorted(missing))}")
        available.add(op.op_id)


def _precedence_bound(rows, external, start):
    """Longest isolated-cost path, including streams and collective rendezvous."""
    ends = {key: max(0, value - start) for key, value in external.items()}
    parents, waiting, successors = {}, {}, {}
    indexed = {row["id"]: row for row in rows}
    for row in rows:
        waiting[row["id"]] = sum(dep not in ends for dep in row["predecessors"])
        for dep in row["predecessors"]:
            successors.setdefault(dep, []).append(row["id"])
    ready = deque(key for key, count in waiting.items() if not count)
    while ready:
        key = ready.popleft()
        row = indexed[key]
        predecessor = max(row["predecessors"], key=ends.__getitem__, default=None)
        parents[key] = predecessor
        ends[key] = ((ends[predecessor] if predecessor is not None else 0)
                     + row.get("precedence_ms", row["isolated_ms"]))
        for successor in successors.get(key, ()):
            waiting[successor] -= 1
            if not waiting[successor]:
                ready.append(successor)
    if len(parents) != len(rows):
        raise ValueError("graph_stream_or_collective_dependency_cycle")
    leaf = max(indexed, key=ends.__getitem__, default=None)
    path, cursor = [], leaf
    while cursor in parents:
        path.append(cursor)
        cursor = parents[cursor]
    return {
        "duration_ms": ends[leaf] if leaf is not None else 0,
        "operators": list(reversed(path)),
        "basis": ("local_dependencies_with_zero_joint_transport_cost"
                  if any("precedence_ms" in row for row in rows)
                  else "isolated_costs_with_data_stream_and_collective_dependencies"),
    }


def _prepare_graphs(case, ops, book, replay_scope, external):
    from .operator_analysis import _replay_scope
    graphs = {r: tuple(book.expand(items)) for r, items in
              (ops.items() if isinstance(ops, Mapping) else ((0, ops),))}
    graphs = bind_cohort_traffic(case, graphs)
    from .sharding.local_tp import lower_local_projection_tp_cohort
    from .sharding.local_tp import local_tp_spans_dp
    if len(graphs) > 1:
        graphs = lower_local_projection_tp_cohort(
            graphs, model=case.model, runtime=case.runtime, machine=case.machine)
    scopes = {r: _replay_scope(items, replay_scope) for r, items in graphs.items()}
    if any(value not in {"none", "full", "piecewise"} for value in scopes.values()):
        raise ValueError("graph_replay_scope_invalid")
    replay_scope = next(iter(set(scopes.values()))) if len(set(scopes.values())) == 1 else "mixed"
    if any(r not in range(case.runtime.parallel.dp) for r in graphs):
        raise ValueError("graph_replica_outside_declared_placement")
    # Effective lowered operators: no timing may price an unmodeled core.
    from .efficiency.unmodeled_gate import assert_timing_supported
    for items in graphs.values():
        assert_timing_supported(case, items, entry="graph_costs")
    for items in graphs.values():
        _validate_graph(items, external)
    groups = cohorts(case.runtime.parallel, case.machine)
    group_for = {r: group for group in groups for r in group}
    for r, items in graphs.items():
        if case.runtime.parallel.ep_spans_dp and any(
            dict(op.metadata).get("group_axis") == "ep" for op in items
        ) and set(group_for[r]) - graphs.keys():
            raise ValueError("graph_expert_collective_requires_complete_cohort")
        if local_tp_spans_dp(case.runtime.parallel) and any(
            dict(op.metadata).get("group_axis") == "local_tp" for op in items
        ) and set(group_for[r]) - graphs.keys():
            raise ValueError("graph_local_tp_collective_requires_complete_cohort")
        if case.runtime.parallel.cp_spans_dp and any(
            dict(op.metadata).get("group_axis") in {"cp", "dcp_dp", "dcp_a2a"} for op in items
        ) and set(group_for[r]) - graphs.keys():
            raise ValueError("graph_cp_collective_requires_complete_cohort")
    return graphs, scopes, replay_scope, groups, group_for


def _make_fast_layer_observer(
    body_key_by_op: Mapping[str, Any],
) -> tuple[Callable[..., None], dict[Any, list[tuple[int, Any, Any]]], list[tuple[Any, Any]]]:
    events_by_layer: dict[Any, list[tuple[int, Any, Any]]] = {}
    non_body_events: list[tuple[Any, Any]] = []

    def _observe(op, replica, _program, ready, complete):
        if replica == 0:
            key = body_key_by_op.get(op.op_id)
            if key is not None:
                events_by_layer.setdefault(key, []).append((int(op.stage), ready, complete))
            else:
                non_body_events.append((ready, complete))

    return _observe, events_by_layer, non_body_events


def _l2_residency_digest(engine) -> str:
    view = getattr(engine, "admission_memory_view", None)
    fn = getattr(view, "residency_fingerprint", None)
    return str(fn()) if callable(fn) else ""


def _run_graph(
    case, ops, graphs, scopes, groups, ctx, book, external, input_ready_ms, *,
    include_submission, trace_limit, fast_span_only, stream_plans, request_views,
    padded_tokens, sequence_layout, timeline_replicas=None, compact=False,
    isolated=False, group_for=None, fast_observer=None,
    l2_fingerprints: dict[str, str] | None = None,
):
    from .runtime.exposed_observer import ExposedObserver

    records, streams = [], {}
    filter_at_observe = timeline_replicas is not None and not compact
    replica_observer = ExposedObserver(
        observed_replicas=timeline_replicas if filter_at_observe else None,
    )
    filtered_non_barrier = 0
    ep_spans = bool(case.runtime.parallel.ep_spans_dp)
    from .sharding.local_tp import local_tp_spans_dp
    ltp_spans = bool(local_tp_spans_dp(case.runtime.parallel))
    cp_spans = bool(case.runtime.parallel.cp_spans_dp)
    cross_dp_axes = set()
    if ep_spans:
        cross_dp_axes.add("ep")
    if ltp_spans:
        cross_dp_axes.add("local_tp")
    if cp_spans:
        cross_dp_axes.update({"cp", "dcp_dp", "dcp_a2a"})
    non_observed_peer_events: list[tuple[Any, int, Any, str]] = []

    def observe(op, replica, program, ready, complete):
        nonlocal filtered_non_barrier
        if filter_at_observe and not replica_observer.should_retain_replica(replica):
            if op.family != "barrier":
                filtered_non_barrier += 1
            if cross_dp_axes:
                meta = dict(op.metadata) if op.metadata else {}
                axis = meta.get("group_axis")
                if axis in cross_dp_axes:
                    if isolated:
                        # Isolated joint-collective pricing needs peer descriptors;
                        # keep only cross-DP collective records when isolated=True.
                        key = _name(replica, op.op_id)
                        records.append({
                            "id": key, "op": op, "replica": replica, "program": program,
                            "predecessors": tuple(_name(replica, dep) for dep in op.deps),
                            "ready": ready, "complete": complete,
                            "collective_algorithm": getattr(book.select(op), "algorithm", None),
                            "_peer_only": True,
                        })
                    else:
                        non_observed_peer_events.append((op, replica, ready, str(axis)))
            return
        key = _name(replica, op.op_id)
        stream = replica, op.stage, op.stream or "main"
        predecessors = [_name(replica, dep) for dep in op.deps]
        if stream in streams:
            predecessors.append(streams[stream])
        streams[stream] = key
        records.append({
            "id": key, "op": op, "replica": replica, "program": program,
            "predecessors": tuple(dict.fromkeys(predecessors)),
            "ready": ready, "complete": complete,
            "collective_algorithm": getattr(book.select(op), "algorithm", None),
        })

    if stream_plans:
        from .streams import validate_stream_plan_ranks
        stream_plans = validate_stream_plan_ranks(stream_plans, set(graphs))
    from .efficiency.cost import CostLedger
    from .implementation_selection import graph_selections
    cost = ctx.cost
    op_count = (sum(len(group) for group in ops.values())
                if isinstance(ops, Mapping) else len(ops))
    execution = Execution(case, cost, recipes=book, bindings=ctx.bindings,
                          ledger=CostLedger(), include_submission=include_submission,
                          on_operator=(fast_observer if fast_span_only else observe),
                          trace_limit=trace_limit, evaluation=ctx,
                          accel_eager_attach=op_count >= EAGER_GRAPH_MIN_OPS,
                          stream_plans=stream_plans)
    # Physical-work coalescing is exact; only an explicit legacy diagnostic run
    # keeps the uncoalesced per-tile shape.
    execution._compact_step = not execution.legacy_diagnostics
    execution.engine._compact_step = execution._compact_step
    execution._fast_span_only = bool(fast_span_only)
    views_source = request_views if request_views is not None else getattr(ops, "request_views", None)
    eff_padded = int(padded_tokens or getattr(ops, "padded_tokens", 0) or 0)
    eff_layout = str(sequence_layout or getattr(ops, "sequence_layout", "") or "")
    execution._active_padded_tokens = eff_padded
    execution._active_layout = eff_layout
    engine = execution.engine
    if l2_fingerprints is not None:
        l2_fingerprints["before"] = _l2_residency_digest(engine)
    engine.now = input_ready_ms
    inputs = {r: {key: engine.submit(
        _name(r, "input:" + key), release=max(value, input_ready_ms),
    ) for key, value in external.items()
        if any(key in op.deps for op in items)} for r, items in graphs.items()}

    def _views_for(replica: int) -> tuple:
        if isinstance(views_source, Mapping):
            return tuple(views_source.get(replica, ()))
        if views_source:
            return tuple(views_source)
        return ()

    works = {r: IterationWork(
        items, IterationPlan(_views_for(r)), case.runtime.parallel.pp,
        sum(v.logical_q_len for v in _views_for(r)),
        eff_padded or sum(v.q_len for v in _views_for(r)),
        None,
        "graph", "declared_graph", replay_scope=scopes[r],
        implementation_selections=graph_selections(items),
    ) for r, items in graphs.items()}
    steps = {}
    if len(works) > 1 and len(groups) > 1:
        from .runtime.dp_cohort_dedup import is_symmetric_dp_cohort
        if is_symmetric_dp_cohort(execution, works, external_inputs=inputs):
            steps.update(execution.add_group(works, input_ready_ms, external_inputs=inputs))
    if not steps:
        for group in groups:
            selected = {r: works[r] for r in group if r in works}
            if len(selected) > 1:
                steps.update(execution.add_group(selected, input_ready_ms, external_inputs=inputs))
            elif selected:
                r, work = next(iter(selected.items()))
                steps[r] = execution.add(work, r, input_ready_ms, external_inputs=inputs[r])
    unique_completions = tuple(dict.fromkeys(step.completion for step in steps.values()))
    engine.advance(until=unique_completions, first=False)
    if l2_fingerprints is not None:
        l2_fingerprints["after"] = _l2_residency_digest(engine)
    skipped_occurrences = filtered_non_barrier
    if not fast_span_only and getattr(execution, "_dedup_cohorts", None):
        from .runtime.dp_cohort_dedup import expand_dedup_graph_artifacts
        skipped_occurrences += expand_dedup_graph_artifacts(
            execution, records, graphs, timeline_replicas=timeline_replicas,
        )
    if non_observed_peer_events and group_for is not None:
        from .topology import Topology
        topology = Topology(case.machine, case.runtime.parallel) if cp_spans else None
        for op, replica, ready, axis in non_observed_peer_events:
            meta = dict(op.metadata) if op.metadata else {}
            if axis == "ep":
                grp = group_for[replica]
            elif axis == "local_tp":
                grp = tuple(meta.get("group_ranks") or group_for[replica])
            else:
                grp = _dcp_replica_subgroup(topology, axis, replica, op.stage, meta, group_for[replica])
            replica_observer.record_peer_readiness((grp, op.op_id), replica, ready.time)
    return execution, engine, records, steps, stream_plans, skipped_occurrences, replica_observer


def _dcp_replica_subgroup(topology, axis, replica, stage, meta, fallback):
    """Map one op's physical DCP collective group to its participating DP replicas."""
    groups = topology.groups(axis, replica, stage, metadata=meta)
    ranks = {int(rk) for group in groups for rk in group}
    if not ranks:
        return fallback
    matched = tuple(
        r for r in fallback
        if any(int(rk) in ranks for rk in topology.ranks(r, stage))
    )
    return matched or fallback


def _bind_collective_readiness(case, records, group_for, execution, replica_observer=None):
    from .sharding.local_tp import local_tp_spans_dp
    from .topology import Topology

    # Aggregate recipes retain their rendezvous. Causal transport plans retain
    # rank entry and round dependencies instead of one all-cohort ready gate.
    joins = {}
    has_ltp = local_tp_spans_dp(case.runtime.parallel)
    has_cp_dp = bool(case.runtime.parallel.cp_spans_dp)
    topology = Topology(case.machine, case.runtime.parallel) if has_cp_dp else None
    if case.runtime.parallel.ep_spans_dp or has_ltp or has_cp_dp:
        for record in records:
            op, r = record["op"], record["replica"]
            meta = dict(op.metadata)
            axis = meta.get("group_axis")
            if case.runtime.parallel.ep_spans_dp and axis == "ep":
                joins.setdefault((group_for[r], op.op_id), []).append(record)
            elif has_ltp and axis == "local_tp":
                grp = tuple(meta.get("group_ranks") or group_for[r])
                joins.setdefault((grp, op.op_id), []).append(record)
            elif has_cp_dp and axis in {"cp", "dcp_dp", "dcp_a2a"}:
                grp = _dcp_replica_subgroup(
                    topology, axis, r, op.stage, meta, group_for[r])
                joins.setdefault((grp, op.op_id), []).append(record)
    for join_key, participants in joins.items():
        if all(item["collective_algorithm"] for item in participants):
            for item in participants:
                item["joint_collective"] = True
            continue
        # A real native cohort owns its own launch-local readiness. Use the
        # actual launch record, not a family label: a plan that merely emits a
        # native-looking op is not evidence that peer work ran.
        if execution.native_group_launches.get(participants[0]["op"].op_id):
            for item in participants:
                item["native_local_ready"] = True
            continue
        dependencies = tuple(dict.fromkeys(dep for item in participants for dep in item["predecessors"]))
        ready_ms = max(item["ready"].time for item in participants)
        if replica_observer is not None:
            peer_ready = replica_observer.collective_peer_ready_ms(join_key)
            if peer_ready is not None and peer_ready > ready_ms:
                ready_ms = peer_ready
        for item in participants:
            item["predecessors"] = dependencies
            item["collective_ready_ms"] = ready_ms

    return joins


def _price_isolated_graph(case, graphs, joins, scopes, book, primitives, ctx, stream_plans):
    from .operator_analysis import cohort_operator_costs, operator_costs, physical_id
    costs, durations, cohort_cache = [], {}, {}
    for participants in joins.values():
        if not participants[0].get("joint_collective"):
            continue
        signature = tuple((item["replica"], item["op"].stage, physical_id(item["op"]))
                          for item in participants)
        if signature not in cohort_cache:
            report = cohort_operator_costs(
                case, {item["replica"]: item["op"] for item in participants},
                recipes=book, primitives=primitives, replay_scope=scopes,
                evaluation=ctx, stream_plans=stream_plans)
            report["id"] = f"cohort{len(cohort_cache)}"
            report["occurrences"] = 0
            for row in report["operators"]:
                row.update(cohort_id=report["id"], occurrences=0, total_ms=0)
                costs.append(row)
            cohort_cache[signature] = report
        report = cohort_cache[signature]
        report["occurrences"] += 1
        for item, row in zip(participants, report["operators"]):
            row["occurrences"] += 1
            row["total_ms"] += row["duration_ms"]
            item["joint_isolated_ms"] = row["duration_ms"]
            item["cohort_id"] = report["id"]
    joint_nodes = {(item["replica"], item["op"].op_id)
                   for members in joins.values() for item in members
                   if item.get("joint_collective")}
    for r, items in graphs.items():
        local = tuple(op for op in items if (r, op.op_id) not in joint_nodes)
        report = operator_costs(case, local, recipes=book, primitives=primitives,
                                replay_scope=scopes[r], replica=r,
                                evaluation=ctx, stream_plans=stream_plans)
        for row in report["operators"]:
            costs.append({"replica": r, **row})
            row_stream = (row["operator"].get("stream") or "main") if stream_plans else ""
            durations[r, row["operator"]["stage"], row["physical_id"], row_stream] = row["duration_ms"]
    isolated_sum = sum(row["total_ms"] for row in costs)
    return costs, durations, isolated_sum, cohort_cache


def _accumulate_compact_record(profile, record, ready_ms, complete_ms, elapsed,
                               metadata, boundary, native_payload,
                               isolated, durations, stream_plans, physical_id):
    op, r = record["op"], record["replica"]
    profile["operator_events"] += 1
    family = profile["by_family"].setdefault(
        op.family, {"operator_events": 0, "elapsed_ms": 0.0, "replicas": {}})
    family["operator_events"] += 1
    family["elapsed_ms"] += elapsed
    rank_totals = profile["by_rank"].setdefault(
        r, {"operator_events": 0, "elapsed_ms": 0.0,
            "ready_ms": ready_ms, "complete_ms": complete_ms})
    rank_totals["operator_events"] += 1
    rank_totals["elapsed_ms"] += elapsed
    rank_totals["ready_ms"] = min(rank_totals["ready_ms"], ready_ms)
    rank_totals["complete_ms"] = max(rank_totals["complete_ms"], complete_ms)
    layer_totals = profile["by_layer"].setdefault(
        op.layer, {"operator_events": 0, "elapsed_ms": 0.0})
    layer_totals["operator_events"] += 1
    layer_totals["elapsed_ms"] += elapsed
    # One canonical operator entry, with each replica's actual declared
    # identity and source boundaries retained below it. No synthetic
    # histogram and no first-rank label leaking onto another rank.
    identity = f"s{op.stage}:{op.op_id}"
    stats = profile["by_operator"].get(identity)
    if stats is None:
        stats = {
            "op_id": op.op_id, "stage": op.stage,
            "operator_events": 0, "ready_ms": ready_ms,
            "complete_ms": complete_ms, "elapsed_ms": 0.0,
            "replicas": {},
        }
        profile["by_operator"][identity] = stats
    stats["operator_events"] += 1
    stats["ready_ms"] = min(stats["ready_ms"], ready_ms)
    stats["complete_ms"] = max(stats["complete_ms"], complete_ms)
    stats["elapsed_sum_ms"] = stats.get("elapsed_sum_ms", 0.0) + elapsed
    stats["elapsed_ms"] = max(stats.get("elapsed_ms", 0.0), elapsed)
    rank = stats["replicas"].get(r)
    if rank is None:
        rank = {
            "operator_events": 0, "ready_ms": ready_ms,
            "complete_ms": complete_ms, "elapsed_ms": 0.0,
            "family": op.family, "layer": op.layer,
            "stream": op.stream or "main",
            "phase": op.phase, "recipe": op.recipe,
            "program": record["program"],
            "source_boundary": op.source_boundary,
        }
        stats["replicas"][r] = rank
    if native_payload is not None:
        rank["native_call"] = native_payload
    rank["operator_events"] += 1
    rank["ready_ms"] = min(rank["ready_ms"], ready_ms)
    rank["complete_ms"] = max(rank["complete_ms"], complete_ms)
    rank["elapsed_ms"] += elapsed
    for key in ("fusion_role", "fusion_group_key", "native_kind"):
        if key in metadata:
            rank[key] = metadata[key]
    rank.update(matrix_wrapper_projection(metadata))
    if boundary:
        rank["native_implementation"] = boundary.get("native")
        rank["physical_source_program"] = (
            metadata.get("fusion_program_profile")
            or metadata.get("fusion_source_profile")
            or boundary.get("program_profile") or boundary.get("traffic_profile"))
    if record.get("joint_collective"):
        rank["collective_algorithm"] = record["collective_algorithm"]
    if record.get("native_local_ready"):
        rank["native_local_ready_ms"] = record["ready"].time
    if "collective_ready_ms" in record:
        wait = ready_ms - record["ready"].time
        rank["local_ready_ms"] = record["ready"].time
        rank["rendezvous_wait_ms"] = max(rank.get("rendezvous_wait_ms", 0.0), wait)
        rank["rendezvous_events"] = rank.get("rendezvous_events", 0) + 1
    if isolated:
        if record.get("joint_collective"):
            iso = record["joint_isolated_ms"]
            record["isolated_ms"] = iso
            record["precedence_ms"] = 0
            rank.update(cohort_id=record["cohort_id"],
                        isolation_boundary="complete_collective_cohort")
        else:
            op_stream = (op.stream or "main") if stream_plans else ""
            iso = (0 if op.family == "barrier"
                   else durations[r, op.stage, physical_id(op), op_stream])
            record["isolated_ms"] = iso
        rank["isolated_ms"] = rank.get("isolated_ms", 0.0) + iso
        stats["isolated_ms"] = stats.get("isolated_ms", 0.0) + iso
        family["isolated_ms"] = family.get("isolated_ms", 0.0) + iso
    family_rank = family["replicas"].setdefault(
        r, {"operator_events": 0, "elapsed_ms": 0.0,
            "ready_ms": ready_ms, "complete_ms": complete_ms})
    family_rank["operator_events"] += 1
    family_rank["elapsed_ms"] += elapsed
    family_rank["ready_ms"] = min(family_rank["ready_ms"], ready_ms)
    family_rank["complete_ms"] = max(family_rank["complete_ms"], complete_ms)


def _graph_operator_rows(records, duration, compact, isolated, durations, stream_plans):
    from .operator_analysis import physical_id
    profile = None
    if compact:
        profile = {
            "duration_ms": duration, "operator_events": 0,
            "by_family": {}, "by_operator": {},
            "by_rank": {}, "by_layer": {},
        }
    rows = []
    for record in records:
        op, r = record["op"], record["replica"]
        ready_ms = record.get("collective_ready_ms", record["ready"].time)
        complete_ms = record["complete"].time
        elapsed = complete_ms - ready_ms
        # Copy declared identities, not nested phase DAGs or guessed name roles.
        metadata = dict(op.metadata)
        boundary = metadata.get("fusion_boundary", {})
        # One normalized actual-invocation descriptor for the per-event row and
        # the compact per-rank projection. It is metadata, never priced work.
        native_call = bind_native_call(op)
        native_payload = native_call.to_dict() if native_call is not None else None
        if compact:
            _accumulate_compact_record(
                profile, record, ready_ms, complete_ms, elapsed,
                metadata, boundary, native_payload,
                isolated, durations, stream_plans, physical_id)
            continue
        row = {
            "id": record["id"], "replica": r, "op_id": op.op_id, "family": op.family,
            "recipe": op.recipe, "program": record["program"], "stage": op.stage,
            "layer": op.layer, "stream": op.stream or "main", "phase": op.phase,
            "source_boundary": op.source_boundary, "predecessors": list(record["predecessors"]),
            "ready_ms": ready_ms, "complete_ms": record["complete"].time,
            "elapsed_ms": record["complete"].time - ready_ms,
        }
        for key in ("fusion_role", "fusion_group_key", "native_kind"):
            if key in metadata:
                row[key] = metadata[key]
        row.update(matrix_wrapper_projection(metadata))
        if native_payload is not None:
            row["native_call"] = native_payload
        if boundary:
            row["native_implementation"] = boundary.get("native")
            row["physical_source_program"] = (
                metadata.get("fusion_program_profile") or metadata.get("fusion_source_profile")
                or boundary.get("program_profile") or boundary.get("traffic_profile"))
        if record.get("joint_collective"):
            row["collective_algorithm"] = record["collective_algorithm"]
        if "collective_ready_ms" in record:
            row["local_ready_ms"] = record["ready"].time
            row["rendezvous_wait_ms"] = ready_ms - record["ready"].time
        if isolated:
            if record.get("joint_collective"):
                row.update(
                    isolated_ms=record["joint_isolated_ms"],
                    precedence_ms=0, cohort_id=record["cohort_id"],
                    isolation_boundary="complete_collective_cohort")
            else:
                op_stream = (op.stream or "main") if stream_plans else ""
                row["isolated_ms"] = (0 if op.family == "barrier"
                                      else durations[r, op.stage, physical_id(op), op_stream])
        rows.append(row)
    return rows, profile


def _isolated_operator_profile(costs, isolated_sum):
    isolated_profile = {
        "operator_sum_ms": isolated_sum,
        "by_family": {}, "by_operator": {},
    }
    for row in costs:
        operator = row["operator"]
        replica = row.get("replica")
        key = (f"r{replica}:{operator['op_id']}" if replica is not None
               else f"{row.get('cohort_id', 'cohort')}:{operator['op_id']}")
        item = isolated_profile["by_operator"].setdefault(key, {
            "replica": replica, "op_id": operator["op_id"],
            "family": operator["family"], "stage": operator["stage"],
            "layer": operator["layer"], "physical_id": row["physical_id"],
            "occurrences": 0, "total_ms": 0.0,
        })
        item["occurrences"] += row["occurrences"]
        item["total_ms"] += row["total_ms"]
        family = isolated_profile["by_family"].setdefault(
            operator["family"], {"occurrences": 0, "total_ms": 0.0})
        family["occurrences"] += row["occurrences"]
        family["total_ms"] += row["total_ms"]
    return isolated_profile


def _record_native_bridge(case, records, execution_summary, profile):
    native_records = [
        record for record in records
        if (record["op"].family in {"mc2_transport", "mc2_protocol"}
            or "mc2_phase" in dict(record["op"].metadata))
    ]
    native_groups = list(execution_summary.get("native_groups", ()) or ())
    if native_records:
        launched_ids = {group["op_id"] for group in native_groups}
        launched = [record for record in native_records
                    if record["op"].op_id in launched_ids]
        protocol = [record for record in native_records
                    if record["op"].op_id not in launched_ids]
        bases = sorted({
            str(basis) for group in native_groups
            for basis in (group.get("payload_basis") or {}).values()})
        source_basis = next(
            (basis for basis in bases if "source_force_eplb" in basis),
            next((basis for basis in bases if basis != "actual_zero"),
                 bases[0] if bases else "unknown"))
        for record in native_records:
            if source_basis != "unknown":
                break
            meta = dict(record["op"].metadata)
            for key in ("source_force_eplb", "source_token_mask"):
                source = meta.get(key)
                if isinstance(source, Mapping) and source.get("basis"):
                    source_basis = str(source["basis"])
                    break
        exact = bool(native_groups) and all(
            basis.startswith("source_force_eplb") or basis == "actual_zero"
            for group in native_groups
            for basis in (group.get("payload_basis") or {}).values())
        expectation = any(
            basis not in {"actual_zero"} and
            not basis.startswith("source_force_eplb")
            for group in native_groups
            for basis in (group.get("payload_basis") or {}).values())
        directed = sum(int(group.get("directed_edges", 0))
                       for group in native_groups)
        tp = int(case.runtime.parallel.tp)
        bridge_status = "actual" if launched else "partial"
        bridge_reason = ("" if launched else
                         "ordinary_frontend_native_group_not_materialized")
        profile["native_bridge"] = {
            "status": bridge_status,
            "reason": bridge_reason,
            "source_basis": source_basis,
            "known_vs_expected": (
                "not_materialized" if not launched else
                "declared_destination_expectation" if expectation else
                "exact_directed_source_frames" if exact and directed else
                "exact_zero_source_frames" if exact else "unknown"),
            "physical_tp_scope": ("resolved" if tp == 1 else
                                  "unresolved_tp_rank_mapping"),
            "native_operators": len(launched),
            "native_launch_groups": len(native_groups),
            "native_kernel_calls": len(native_groups),
            "launch_modes": sorted({
                str(group.get("launch_mode", ""))
                for group in native_groups if group.get("launch_mode")}),
            "protocol_operators": len(protocol),
            "directed_edges": directed,
            "directed_frames": sum(int(group.get("directed_frames", 0))
                                   for group in native_groups),
            "transport_tx_bytes": sum(float(group.get(
                "transport_tx_bytes", 0.0)) for group in native_groups),
            "transport_rx_bytes": sum(float(group.get(
                "transport_rx_bytes", 0.0)) for group in native_groups),
            "transport_frames": sum(int(group.get(
                "transport_frames", 0)) for group in native_groups),
            "transport_edges": sum(int(group.get(
                "transport_edges", 0)) for group in native_groups),
            "total_edges": sum(int(group.get(
                "total_edges", 0)) for group in native_groups),
            "payload_edges": sum(int(group.get(
                "payload_edges", 0)) for group in native_groups),
            "count_edges": sum(int(group.get(
                "count_edges", 0)) for group in native_groups),
            "operators": native_groups,
        }


def _assemble_graph_result(
    case, graphs, records, steps, execution, engine, rows, profile,
    resources, head, *, compact, isolated, costs, isolated_sum,
    isolated_profile, cohort_cache, external_times, input_ready_ms,
    duration, audit_allocations,
):
    execution_summary = execution.summary()
    parameter_summary = execution.cost.summary()
    if compact:
        result = {
            **head, "resources": resources,
            "execution": execution_summary, "parameters": parameter_summary,
        }
        profile["steps_by_replica"] = {
            replica: {
                "start_ms": step.start_ms,
                "device_end_ms": step.device_complete.time,
                "end_ms": step.completion.time,
            }
            for replica, step in steps.items()
        }
        _record_native_bridge(case, records, execution_summary, profile)
        result.update(
            result_mode="compact", profile=profile,
            event_profile=bounded_event_profile(engine),
            timeline_omitted=True, events_omitted=True,
        )
    else:
        # Keep the historical detailed key order byte-for-byte.
        result = {
            **head, "timeline": rows, "resources": resources,
            "execution": execution_summary, "parameters": parameter_summary,
            "events": list(engine.trace),
        }
        evicted_spans = getattr(engine, "_evicted_service_spans", None)
        if evicted_spans:
            result["evicted_service_spans"] = {
                key: tuple(spans) for key, spans in evicted_spans.items()
            }
    if graphs:
        observed_reps = {record["replica"] for record in records if not record.get("_peer_only")}
        exposed_rep = min(graphs) if (not observed_reps or min(graphs) in observed_reps) else min(observed_reps)
        result.update(_exposed_breakdown(
            records, duration, input_ready_ms, exposed_rep,
            stage_rates_by_key=_stage_role_rates(execution)))
    else:
        result.update(unavailable_breakdown(
            "exposed_operator_intervals_missing", boundary="device_step"))
    if isolated:
        path = _precedence_bound(records if compact else rows,
                                 external_times, input_ready_ms)
        capacity_bound = max((row["capacity_bound_ms"] for row in resources), default=0)
        result["isolated"] = {
            "boundary": "sum_of_isolated_device_programs",
            "operator_sum_ms": isolated_sum,
            **({"by_family": isolated_profile["by_family"],
                "by_operator": isolated_profile["by_operator"],
                "operators_omitted": True}
               if compact else {"operators": costs}),
            **({"collective_cohorts": [
                {key: value for key, value in report.items() if key != "operators"}
                for report in cohort_cache.values()
            ], "sum_interpretation": "sum_of_replica_representative_completion_spans"}
               if cohort_cache else {}),
        }
        result["bounds"] = {
            "precedence": path, "resource_capacity_ms": capacity_bound,
            "combined_ms": max(path["duration_ms"], capacity_bound),
            "interpretation": "lower_bounds_for_this_declared_graph_not_optimization_savings",
        }
    if audit_allocations:
        # Bounded real grant/release evidence for shared-pool admission.
        result["allocation_trace"] = tuple(getattr(engine, "allocation_trace", ()))
    return result


def graph_costs(case, ops, *, recipes=None, primitives=None, external_inputs=None,
                input_ready_ms=0, include_submission=False, replay_scope=None,
                isolated=True, trace_limit=0, compact=False,
                audit_allocations=False, evaluation=None,
                fast_span_only=False, stream_plans=None,
                request_views=None, padded_tokens=0, sequence_layout="",
                timeline_replicas=None, layer_span_body_ids=None):
    """Run a declared DAG or replica->DAG mapping without a request scheduler.

    External input keys are operator dependency names and values are absolute
    virtual ready times, shared across selected replicas. A sequence selects
    replica 0. Cross-DP expert collectives require their complete cohort mapping.

    ``compact=True`` is additive: it omits the per-event ``timeline`` and
    ``events`` lists and returns the same duration, operator events and
    readiness/completion boundaries as bounded ``profile``/``event_profile``
    reductions. No synthetic time histogram is built; each replica's declared
    identity and source ready/complete boundaries are retained.

    This is the generic execution-graph entry: sampling, token commit and
    acceptance are ordinary declared nodes here. A pure-model business entry
    excludes them by selecting an assembly scope, never by a family blacklist.
    """
    from .body_defaults import consume_target_declaration
    case = consume_target_declaration(parse_case(case), ops)
    if input_ready_ms < 0 or not math.isfinite(input_ready_ms):
        raise ValueError("graph_input_ready_time_must_be_finite_and_nonnegative")
    # A cut dependency is a declared input, never an implicit body default.
    # Callers that intentionally cut a body graph provide the boundary
    # explicitly (efficiency/body entry points pass ``{"analysis_input": 0}``);
    # an undeclared cut raises ``graph_dependency_unavailable`` below.
    external = {} if external_inputs is None else dict(external_inputs)
    if any(value < 0 or not math.isfinite(value) for value in external.values()):
        raise ValueError("graph_external_input_time_must_be_finite_and_nonnegative")
    from .evaluation import EvaluationContext, context_for
    ctx = (context_for(case, evaluation, recipes) if evaluation is not None
           else EvaluationContext.create(case, recipes, primitives))
    book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    graphs, scopes, replay_scope, groups, group_for = _prepare_graphs(
        case, ops, book, replay_scope, external)
    external_times = {_name(r, key): max(value, input_ready_ms)
                      for r in graphs for key, value in external.items()}
    keep_replicas = (
        frozenset(int(r) for r in timeline_replicas)
        if timeline_replicas is not None else None
    )
    fast_observer, fast_layer_events, fast_non_body_events = (
        _make_fast_layer_observer(
            layer_span_body_ids if isinstance(layer_span_body_ids, Mapping)
            else {oid: ("target", -1) for oid in layer_span_body_ids}
        )
        if (fast_span_only and layer_span_body_ids) else (None, None, None)
    )
    l2_fingerprints = {} if layer_span_body_ids is not None else None
    execution, engine, records, steps, stream_plans, skipped_occurrences, replica_observer = _run_graph(
        case, ops, graphs, scopes, groups, ctx, book, external, input_ready_ms,
        include_submission=include_submission, trace_limit=trace_limit,
        fast_span_only=fast_span_only, stream_plans=stream_plans,
        request_views=request_views, padded_tokens=padded_tokens, sequence_layout=sequence_layout,
        timeline_replicas=keep_replicas, compact=compact, isolated=isolated,
        group_for=group_for, fast_observer=fast_observer,
        l2_fingerprints=l2_fingerprints)
    end = max((step.completion.time for step in steps.values()), default=input_ready_ms)
    duration = end - input_ready_ms
    if fast_span_only:
        # Same Engine, same declared work, same virtual timestamps: only the
        # per-operator timeline and the isolated pricing pass are omitted.
        summary = execution.summary()
        out_fast = {
            "schema": "mindie-sim.graph/1", **model_coverage(case.model),
            "boundary": ("submitted_device_graph" if include_submission
                         else "device_graph"),
            "input_ready_ms": input_ready_ms, "complete_ms": end,
            "duration_ms": duration, "replicas": list(graphs),
            "replay_scope": replay_scope,
            "operator_occurrences": 0, "resources": [],
            "execution": summary,
            "engine_backend": summary.get("accel_backend"),
            "accel_status": accel_status(engine.accel_status()),
            "fast_span_only": True,
        }
        if fast_layer_events is not None:
            out_fast["layer_bounds"] = {
                lyr: (
                    evs[0][0],
                    min(float(r.time) for _, r, _ in evs),
                    max(float(c.time) for _, _, c in evs),
                )
                for lyr, evs in fast_layer_events.items() if evs
            }
            out_fast["non_body_bounds"] = [
                (float(r.time), float(c.time))
                for r, c in (fast_non_body_events or ())
            ]
        if l2_fingerprints is not None:
            out_fast["l2_residency_fingerprints"] = (
                l2_fingerprints.get("before", ""),
                l2_fingerprints.get("after", ""),
            )
        return out_fast
    joins = _bind_collective_readiness(case, records, group_for, execution, replica_observer=replica_observer)
    costs, durations, isolated_sum, cohort_cache = [], {}, 0.0, {}
    if isolated:
        costs, durations, isolated_sum, cohort_cache = _price_isolated_graph(
            case, graphs, joins, scopes, book, primitives, ctx, stream_plans)
    row_records = (
        [record for record in records if not record.get("_peer_only") and record["replica"] in keep_replicas]
        if (keep_replicas is not None and not compact) else records
    )
    rows, profile = _graph_operator_rows(
        row_records, duration, compact, isolated, durations, stream_plans)
    isolated_profile = None
    if compact and isolated:
        isolated_profile = _isolated_operator_profile(costs, isolated_sum)
        costs = []
    resources = [{
        "resource": name, "work": amount, "unit": engine.resources[name].unit,
        "capacity_per_ms": engine.resources[name].rate,
        "capacity_bound_ms": amount / engine.resources[name].rate,
        "busy_ms": engine.busy_ms[name],
    } for name, amount in engine.serviced.items()
        if amount and "stream_quota@" not in name]
    resources.sort(key=lambda row: row["capacity_bound_ms"], reverse=True)
    accel_view = engine.accel_status()
    head = {
        "schema": "mindie-sim.graph/1",
        **model_coverage(case.model),
        "engine_backend": accel_view.get("accel_backend"),
        "accel_status": accel_status(accel_view),
        "boundary": "submitted_device_graph" if include_submission else "device_graph",
        "input_ready_ms": input_ready_ms, "complete_ms": end, "duration_ms": duration,
        "replicas": list(graphs), "replay_scope": replay_scope,
        "replay_scope_by_replica": scopes,
        "operator_occurrences": sum(
            record["op"].family != "barrier" and not record.get("_peer_only")
            for record in records) + skipped_occurrences,
    }
    assembled = _assemble_graph_result(
        case, graphs, records, steps, execution, engine, rows, profile,
        resources, head, compact=compact, isolated=isolated, costs=costs,
        isolated_sum=isolated_sum, isolated_profile=isolated_profile,
        cohort_cache=cohort_cache, external_times=external_times,
        input_ready_ms=input_ready_ms, duration=duration,
        audit_allocations=audit_allocations,
    )
    if l2_fingerprints is not None:
        assembled["l2_residency_fingerprints"] = (
            l2_fingerprints.get("before", ""),
            l2_fingerprints.get("after", ""),
        )
    return assembled
