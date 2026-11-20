"""Metadata normalization, cache keys, and N-stream owner/FIFO helpers.

[Role]
  Provide canonical layer-normalized operator metadata keys, bounded metadata
  diffs/patches, rank-token normalization, and N-stream (``N >= 1``) stream
  owner application helpers for ``mindie_sim.prediction.execution``.

[Invariants]
  * Metadata caches are strictly bounded by ``META_CACHE_LIMIT`` /
    ``NO_REV_OPS_CACHE_LIMIT`` and cache only immutable structural projections,
    never simulated durations or served counts.
  * Stream helpers impose no fixed two-stream ceiling and preserve per-stream
    ordering across arbitrary ``N >= 1`` declared stream identifiers.

[Timeouts & Locks]
  Bounded in-memory dictionary lookups and structural projections; no I/O.

[Side-Effect Policy]
  Maintains bounded process-local structural key caches only.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from dataclasses import dataclass
from fractions import Fraction
import json
import re
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError

META_CACHE_LIMIT = 32768
NO_REV_OPS_CACHE_LIMIT = 256
DEFAULT_STREAM_ID = "main"

_LAYER_PREFIX_RE = re.compile(r"\bs\d+\.l\d+\.")
_LAYER_INDEX_RE = re.compile(r'"layer":\s*\d+')

_DIAGNOSTIC_ONLY_META_KEYS = frozenset({
    "structure_subgraphs", "fusion_semantic_ids", "fusion_external_outputs",
    "layer", "gdn_state_contract", "gdn_state_effects",
    "state_candidate_stores", "state_value_outputs",
    "state_accesses", "memory_accesses",
    "conv_state_effects", "kda_state_effects",
})
_SCAN_REBIND_KEYS = frozenset({
    "state_accesses", "memory_accesses", "memory_bindings", "memory_binding",
})
_SCAN_MEM_BIND_KEYS = frozenset({"memory_bindings", "memory_binding"})

_META_TEXT_CACHE: dict = {}
_META_ID_CACHE: dict = {}
_META_KEY_CACHE: dict = {}
_META_DICT_CACHE: dict = {}
_META_SCAN_CACHE: dict = {}
_NO_REV_OPS_CACHE: dict = {}
_DECLARATION_CACHE: dict = {}
_DEVICE_UNITS_CACHE: dict = {}


def _freeze_key(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_key(item) for item in value)
    if isinstance(value, list):
        return ("__list__", tuple(_freeze_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("__set__", tuple(sorted((_freeze_key(item) for item in value), key=repr)))
    if isinstance(value, _AbcMapping):
        return ("__map__", tuple(sorted(
            ((str(k), _freeze_key(v)) for k, v in value.items()),
            key=lambda pair: pair[0],
        )))
    try:
        hash(value)
        return value
    except TypeError:
        return ("__repr__", type(value), repr(value))


def meta_scan_flags(metadata) -> tuple[bool, bool, bool]:
    """Return cached ``(has_rebind_keys, has_reverse_of_collective, has_mem_bind_keys)``."""
    if not metadata:
        return False, False, False
    hit = _META_SCAN_CACHE.get(id(metadata))
    if hit is not None and hit[0] is metadata:
        return hit[1], hit[2], hit[3]
    if isinstance(metadata, _AbcMapping):
        has_reb = any(k in _SCAN_REBIND_KEYS for k in metadata)
        has_rev = bool(metadata.get("reverse_of_collective"))
        has_mb = any(k in _SCAN_MEM_BIND_KEYS for k in metadata)
    else:
        has_reb = False
        has_rev = False
        has_mb = False
        for k, v in metadata:
            if not has_reb and k in _SCAN_REBIND_KEYS:
                has_reb = True
            if not has_mb and k in _SCAN_MEM_BIND_KEYS:
                has_mb = True
            if k == "reverse_of_collective" and v:
                has_rev = True
    if len(_META_SCAN_CACHE) >= META_CACHE_LIMIT:
        _META_SCAN_CACHE.clear()
    _META_SCAN_CACHE[id(metadata)] = (metadata, has_reb, has_rev, has_mb)
    return has_reb, has_rev, has_mb


def mark_no_rev_collective(ops: tuple) -> None:
    if len(_NO_REV_OPS_CACHE) >= NO_REV_OPS_CACHE_LIMIT:
        _NO_REV_OPS_CACHE.clear()
    _NO_REV_OPS_CACHE[id(ops)] = ops


def seed_meta_id_key(metadata, key) -> None:
    if len(_META_ID_CACHE) >= META_CACHE_LIMIT:
        _META_ID_CACHE.clear()
    _META_ID_CACHE[id(metadata)] = (metadata, key)


def inherit_meta_id_key(old_meta, new_meta) -> None:
    hit = _META_ID_CACHE.get(id(old_meta))
    if hit is not None and hit[0] is old_meta:
        if len(_META_ID_CACHE) >= META_CACHE_LIMIT:
            _META_ID_CACHE.clear()
        _META_ID_CACHE[id(new_meta)] = (new_meta, hit[1])


def _metadata_dict(op) -> dict:
    """Cached ``dict(op.metadata)`` for a repeated operator declaration."""
    metadata = op.metadata
    try:
        cached = _META_DICT_CACHE.get(metadata)
    except TypeError:
        return dict(metadata)
    if cached is None:
        cached = dict(metadata)
        if len(_META_DICT_CACHE) >= META_CACHE_LIMIT:
            _META_DICT_CACHE.clear()
        _META_DICT_CACHE[metadata] = cached
    return cached


def _declaration_of(op):
    """Cached kernel-group declaration for a repeated operator declaration."""
    metadata = op.metadata
    try:
        cached = _DECLARATION_CACHE.get(metadata)
    except TypeError:
        from ..kernel_groups import declaration
        return declaration(op)
    if cached is None:
        from ..kernel_groups import declaration
        cached = declaration(op)
        if len(_DECLARATION_CACHE) >= META_CACHE_LIMIT:
            _DECLARATION_CACHE.clear()
        _DECLARATION_CACHE[metadata] = cached
    return cached


def _project_meta_pair(pair):
    k, v = pair
    if k == "conv_state_contract" and isinstance(v, _AbcMapping):
        return (
            k,
            (
                bool(v.get("read_records")),
                bool(v.get("write_records")),
                len(v.get("commit_records", ())),
                len(v.get("rollback_records", ())),
            ),
        )
    return pair


def op_meta_key_from_metadata(metadata) -> Any:
    """Layer-zero canonical projection of one immutable metadata declaration."""
    from ..physical_key import operator_metadata_key
    hit = _META_KEY_CACHE.get(id(metadata))
    if hit is not None and hit[0] is metadata:
        return hit[1]
    key = operator_metadata_key(metadata, 0, 0)
    if len(_META_KEY_CACHE) >= META_CACHE_LIMIT:
        _META_KEY_CACHE.clear()
    _META_KEY_CACHE[id(metadata)] = (metadata, key)
    return key


def _legacy_op_meta_key_from_metadata(metadata) -> Any:
    id_hit = _META_ID_CACHE.get(id(metadata))
    if id_hit is not None and id_hit[0] is metadata:
        return id_hit[1]
    items = metadata.items() if isinstance(metadata, _AbcMapping) else (metadata or ())
    filtered = tuple(
        _project_meta_pair(pair) for pair in items
        if pair[0] not in _DIAGNOSTIC_ONLY_META_KEYS
    )
    hashable_filtered = True
    try:
        cached = _META_TEXT_CACHE.get(filtered)
    except TypeError:
        cached = None
        hashable_filtered = False
    if cached is not None:
        if len(_META_ID_CACHE) >= META_CACHE_LIMIT:
            _META_ID_CACHE.clear()
        _META_ID_CACHE[id(metadata)] = (metadata, cached)
        return cached
    try:
        meta_dict = dict(filtered)
        text = json.dumps(meta_dict, sort_keys=True, default=repr,
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        key = _op_meta_key_slow_meta(metadata)
        if len(_META_ID_CACHE) >= META_CACHE_LIMIT:
            _META_ID_CACHE.clear()
        _META_ID_CACHE[id(metadata)] = (metadata, key)
        return key
    key = ("__json__", _LAYER_INDEX_RE.sub('"layer":0',
                                           _LAYER_PREFIX_RE.sub("s_.l_.", text)))
    if hashable_filtered:
        if len(_META_TEXT_CACHE) >= META_CACHE_LIMIT:
            _META_TEXT_CACHE.clear()
        _META_TEXT_CACHE[filtered] = key
    if len(_META_ID_CACHE) >= META_CACHE_LIMIT:
        _META_ID_CACHE.clear()
    _META_ID_CACHE[id(metadata)] = (metadata, key)
    return key


def _op_meta_key(op) -> Any:
    """Canonical occurrence-relative projection of one operator's metadata."""
    from ..physical_key import operator_metadata_key
    return operator_metadata_key(op.metadata, op.stage, op.layer)


def _op_meta_key_slow_meta(metadata) -> Any:
    def _norm(value: Any) -> Any:
        if isinstance(value, str):
            return _LAYER_PREFIX_RE.sub("s_.l_.", value)
        if isinstance(value, (int, float, bool, bytes, type(None))):
            return value
        if isinstance(value, tuple):
            return tuple(_norm(item) for item in value)
        if isinstance(value, list):
            return ("__list__", tuple(_norm(item) for item in value))
        if isinstance(value, (set, frozenset)):
            return ("__set__", tuple(sorted((_norm(item) for item in value), key=repr)))
        if isinstance(value, _AbcMapping):
            return ("__map__", tuple(sorted(
                ((str(k), _norm(_project_meta_pair((k, v))[1]))
                 for k, v in value.items()
                 if k not in _DIAGNOSTIC_ONLY_META_KEYS),
                key=lambda pair: pair[0],
            )))
        try:
            hash(value)
            return value
        except TypeError:
            return ("__repr__", type(value), repr(value))

    return _norm(dict(metadata))


def _op_meta_key_slow(op) -> Any:
    return _op_meta_key_slow_meta(op.metadata)


def _rank_token(context) -> str | None:
    """Replica marker embedded in physical resource names, such as ``@r3``."""
    for name in context.resource_names.values():
        start = name.find("@r")
        if start < 0:
            continue
        end = start + 2
        while end < len(name) and name[end].isdigit():
            end += 1
        if end > start + 2:
            return name[start:end]
    return None


def _collective_native_basis(op, group_size):
    """Logical per-rank native input admitted by a TP/DP collective op."""
    metadata = dict(op.metadata)
    collective = str(metadata.get("collective", op.family)).lower()
    collective = {"allgather": "all_gather", "reducescatter": "reduce_scatter",
                  "alltoall": "all_to_all"}.get(collective, collective)
    count = max(1, int(group_size) - 1)
    if collective == "all_gather" and op.link_bytes > 0:
        expected = Fraction(str(op.link_bytes)) / count
    elif collective == "reduce_scatter" and op.link_bytes > 0:
        expected = Fraction(str(op.link_bytes * group_size)) / count
    else:
        expected = Fraction(op.link_bytes) if op.link_bytes > 0 else None
    dtype = (metadata.get("payload_dtype") or metadata.get("dtype")
             or metadata.get("activation_dtype") or op.dtype or None)
    return collective, expected, dtype


def _normalize_token(value, token: str):
    if isinstance(value, str):
        return value.replace(token, "@r*")
    if isinstance(value, tuple):
        return tuple(_normalize_token(item, token) for item in value)
    if isinstance(value, list):
        return [_normalize_token(item, token) for item in value]
    if isinstance(value, _AbcMapping):
        return {str(key): _normalize_token(item, token) for key, item in value.items()}
    return value


def _isolated_script_key(prog_key, context, token: str):
    """Physical identity shared by equal ranks. Replica ids are not part of it."""
    rates = tuple(sorted(
        (role, context.rates[name])
        for role, name in context.resource_names.items()
        if name in context.rates
    ))
    pools = tuple(sorted(
        (str(key), float(value)) for key, value in context.pool_capacities.items()
    ))
    return (_normalize_token(prog_key[1:], token), rates, pools, int(context.core_count))


def _meta_diff(orig: Any, res: Any) -> Any:
    if isinstance(orig, _AbcMapping) and isinstance(res, _AbcMapping):
        removed = frozenset(orig.keys() - res.keys())
        updated = {}
        for k, v in res.items():
            if k in orig:
                if orig[k] != v:
                    updated[k] = _meta_diff(orig[k], v)
            elif k == "state_contract" and "gdn_state_contract" in orig and orig["gdn_state_contract"] == v:
                updated[k] = ("__copy_key__", "gdn_state_contract")
            elif k == "state_contract" and "conv_state_contract" in orig and orig["conv_state_contract"] == v:
                updated[k] = ("__copy_key__", "conv_state_contract")
            elif k == "state_effects" and "gdn_state_effects" in orig and orig["gdn_state_effects"] == v:
                updated[k] = ("__copy_key__", "gdn_state_effects")
            elif k == "state_effects" and "conv_state_effects" in orig and orig["conv_state_effects"] == v:
                updated[k] = ("__copy_key__", "conv_state_effects")
            else:
                updated[k] = ("__set__", v)
        return ("__dict__", removed, updated)
    if (isinstance(orig, tuple) and isinstance(res, tuple) and len(orig) == len(res)
            and orig and all(isinstance(x, _AbcMapping) for x in orig)
            and all(isinstance(x, _AbcMapping) for x in res)):
        return ("__tuple_map__", tuple(_meta_diff(a, b) for a, b in zip(orig, res)))
    return ("__set__", res)


def _meta_patch(orig: Any, diff: Any) -> Any:
    kind = diff[0]
    if kind == "__set__":
        return diff[1]
    if kind == "__dict__" and isinstance(orig, _AbcMapping):
        _, removed, updated = diff
        out = {k: v for k, v in orig.items() if k not in removed}
        for k, sub in updated.items():
            if sub[0] == "__copy_key__":
                if sub[1] in orig:
                    out[k] = orig[sub[1]]
            else:
                out[k] = _meta_patch(out[k], sub) if k in out else sub[1]
        return out
    if kind == "__tuple_map__" and isinstance(orig, tuple) and len(orig) == len(diff[1]):
        return tuple(_meta_patch(a, sub) for a, sub in zip(orig, diff[1]))
    return diff[-1]


def stream_id_for_op(op) -> str:
    """Resolve canonical N-stream identifier for an operator (``N >= 1``)."""
    return str(getattr(op, "stream", None) or DEFAULT_STREAM_ID)


def apply_stream_owners(stream_quota_by_res, context, program):
    """Apply declared N-stream owner quotas and placements to a lowered program."""
    owners = context.stream_owners
    if owners is None or context.datapath is None:
        return program
    from ..streams import (
        MAIN_STREAM,
        apply_stream_owners_to_program,
        check_work_against_stream_budget,
    )
    stream_id = getattr(context.op, "stream", None) or MAIN_STREAM
    check_work_against_stream_budget(
        context.work,
        owners,
        resource_names=context.resource_names,
        stream_id=stream_id,
        label=str(getattr(context.op, "op_id", "op")),
    )
    quota_map = {
        phys: stream_quota_by_res[(phys, stream_id)]
        for phys in context.resource_names.values()
        if (phys, stream_id) in stream_quota_by_res
    }
    return apply_stream_owners_to_program(
        program,
        owners,
        context.datapath,
        resource_names=context.resource_names,
        resource_units=context.resource_units,
        rates=context.rates,
        stream_id=stream_id,
        quota_resources=quota_map or None,
        pool_names=context.pool_names,
        pool_units=context.pool_units,
    )


class ResourceRates(Mapping):
    """Read-only live rate view; PD extends the same Engine with new namespaces."""

    def __init__(self, resources):
        self.resources = resources

    def __getitem__(self, key):
        return self.resources[key].rate

    def __iter__(self):
        return iter(self.resources)

    def __len__(self):
        return len(self.resources)


@dataclass(frozen=True)
class _ServiceSetupProgram:
    """One declared resource-service fixed term before an existing body."""

    body: Any
    setup_ms: float

    def start(self, engine, name, ready, *, owner=""):
        setup = engine.submit(name + ":service_setup", after=(ready,),
                              latency=self.setup_ms, owner=owner)
        return self.body.start(engine, name, ready=setup, owner=owner)


class _LedgerBoundCost:
    """CostModel view whose diagnostic records belong to one execution."""

    def __init__(self, model, ledger):
        self.model, self.ledger = model, ledger

    @property
    def case(self):
        return self.model.case

    @property
    def primitives(self):
        return self.model.primitives

    @property
    def counts(self):
        return self.ledger.counts

    @property
    def services(self):
        return self.ledger.services

    @property
    def mappings(self):
        return self.ledger.mappings

    @property
    def invocations(self):
        return self.ledger.invocations

    @property
    def regimes(self):
        return self.ledger.regimes

    @property
    def joint_native(self):
        return self.ledger.joint_native

    @property
    def diagnostic_primitive_lookups(self):
        return self.ledger.diagnostic_primitive_lookups

    @property
    def diagnostic_primitive_hits(self):
        return self.ledger.diagnostic_primitive_hits

    def price(self, op, scope):
        return self.model.price(op, scope, ledger=self.ledger)

    def lookup_diagnostic(self, op, scope):
        return self.model.lookup_diagnostic(op, scope, ledger=self.ledger)

    def record_mapping(self, mapping):
        return self.model.record_mapping(mapping, ledger=self.ledger)

    def record_service(self, curve, basis, *, capacity_limited=False):
        return self.model.record_service(
            curve, basis, capacity_limited=capacity_limited, ledger=self.ledger)

    def record_invocation(self, record):
        return self.model.record_invocation(record, ledger=self.ledger)

    def record_regime(self, *args, **kwargs):
        return self.model.record_regime(*args, ledger=self.ledger, **kwargs)

    def record_joint_native(self, selection, reference_cost):
        return self.model.record_joint_native(
            selection, reference_cost, ledger=self.ledger)

    def summary(self, ledger=None):
        return self.model.summary(self.ledger if ledger is None else ledger)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _ledger_recorder(method, ledger):
    """Bind one CostModel recorder to this execution's private ledger."""
    def record(*args, **kwargs):
        return method(*args, ledger=ledger, **kwargs)
    return record


@dataclass(frozen=True)
class _CollectivePlanBinding:
    """One selected collective algorithm and its lazily built topology plan."""

    algorithm: str
    plan: object | None = None
    factory: object | None = None

    def __call__(self):
        if self.plan is not None:
            return self.plan
        if self.factory is None:
            raise ModelingError("collective_plan_factory_missing")
        return self.factory()

