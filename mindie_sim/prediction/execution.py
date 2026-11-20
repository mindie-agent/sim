"""Compile iteration DAGs onto the shared physical execution engine.

[Role] Bind rank placement, host submission, kernel recipes and device streams.
[Invariants] Model names do not select execution. Same-stream order and data
  dependencies are distinct from resource contention. Host submission can run
  ahead of device completion. Link work is never reduced by overlap.
[Timeouts & Locks] Finite virtual time; at most active iteration DAGs are live.
[Side-Effect Policy] Mutates only the simulation Engine and aggregate counters.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
import json
import re
from collections import deque
from collections.abc import Mapping as _AbcMapping
from typing import Any, Mapping

from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from .cost import CostModel
from .cohort_traffic import bind_cohort_traffic
from .collectives import plan_options as collective_options
from .engine import Engine, Event, Resource
from .execution_state import (
    bind_op_accesses, mandatory_edges, memory_bindings_for,
)
from .mechanism import Geometry, Work
from .program import KernelProgram, RecipeBook, RecipeContext
from .topology import Topology
from .serving_io import ServingIO
from .step_metrics import StepSummary


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
        return ("__map__", tuple(sorted(((str(k), _freeze_key(v)) for k, v in value.items()), key=lambda pair: pair[0])))
    try:
        hash(value)
        return value
    except TypeError:
        return ("__repr__", type(value), repr(value))


_LAYER_PREFIX_RE = re.compile(r"\bs\d+\.l\d+\.")
#: Metadata that describes an occurrence, not the physical program it lowers.
#: Fusion scopes carry the layer ordinal next to the op id; the lowered
#: program, its work and its allocations are layer-independent, so two layers
#: with the same operator geometry share one prepared lowering.
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

_LAYER_INDEX_RE = re.compile(r'"layer":\s*\d+')

#: Pure structural projections of one operator declaration.  The lowering
#: identity is a function of ``op.metadata`` alone: two body occurrences of the
#: same layer-normalized operator geometry therefore share one entry.  Never a
#: duration, a schedule or a served amount - only the declaration text.
_META_TEXT_CACHE: dict = {}
_META_ID_CACHE: dict = {}
_META_KEY_CACHE: dict = {}
_META_DICT_CACHE: dict = {}
_META_SCAN_CACHE: dict = {}
_NO_REV_OPS_CACHE: dict = {}
_DECLARATION_CACHE: dict = {}
_DEVICE_UNITS_CACHE: dict = {}
_META_CACHE_LIMIT = 32768
ZERO = 0
ONE = 1
FIRST = 0
SECOND = 1
HOST_LOOKAHEAD_DEFAULT = 8


def meta_scan_flags(metadata) -> tuple[bool, bool, bool]:
    """Return cached ``(has_rebind_keys, has_reverse_of_collective, has_mem_bind_keys)``."""
    if not metadata:
        return False, False, False
    hit = _META_SCAN_CACHE.get(id(metadata))
    if hit is not None and hit[FIRST] is metadata:
        return hit[SECOND], hit[2], hit[3]
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
    if len(_META_SCAN_CACHE) >= _META_CACHE_LIMIT:
        _META_SCAN_CACHE.clear()
    _META_SCAN_CACHE[id(metadata)] = (metadata, has_reb, has_rev, has_mb)
    return has_reb, has_rev, has_mb


def mark_no_rev_collective(ops: tuple) -> None:
    if len(_NO_REV_OPS_CACHE) >= 256:
        _NO_REV_OPS_CACHE.clear()
    _NO_REV_OPS_CACHE[id(ops)] = ops


def seed_meta_id_key(metadata, key) -> None:
    if len(_META_ID_CACHE) >= _META_CACHE_LIMIT:
        _META_ID_CACHE.clear()
    _META_ID_CACHE[id(metadata)] = (metadata, key)


def inherit_meta_id_key(old_meta, new_meta) -> None:
    hit = _META_ID_CACHE.get(id(old_meta))
    if hit is not None and hit[0] is old_meta:
        if len(_META_ID_CACHE) >= _META_CACHE_LIMIT:
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
        if len(_META_DICT_CACHE) >= _META_CACHE_LIMIT:
            _META_DICT_CACHE.clear()
        _META_DICT_CACHE[metadata] = cached
    return cached


def _declaration_of(op):
    """Cached kernel-group declaration for a repeated operator declaration."""
    metadata = op.metadata
    try:
        cached = _DECLARATION_CACHE.get(metadata)
    except TypeError:
        from .kernel_groups import declaration
        return declaration(op)
    if cached is None:
        from .kernel_groups import declaration
        cached = declaration(op)
        if len(_DECLARATION_CACHE) >= _META_CACHE_LIMIT:
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
    from .physical_key import operator_metadata_key
    hit = _META_KEY_CACHE.get(id(metadata))
    if hit is not None and hit[FIRST] is metadata:
        return hit[SECOND]
    key = operator_metadata_key(metadata, ZERO, ZERO)
    if len(_META_KEY_CACHE) >= _META_CACHE_LIMIT:
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
        if len(_META_ID_CACHE) >= _META_CACHE_LIMIT:
            _META_ID_CACHE.clear()
        _META_ID_CACHE[id(metadata)] = (metadata, cached)
        return cached
    try:
        meta_dict = dict(filtered)
        text = json.dumps(meta_dict, sort_keys=True, default=repr,
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        key = _op_meta_key_slow_meta(metadata)
        if len(_META_ID_CACHE) >= _META_CACHE_LIMIT:
            _META_ID_CACHE.clear()
        _META_ID_CACHE[id(metadata)] = (metadata, key)
        return key
    key = ("__json__", _LAYER_INDEX_RE.sub('"layer":0',
                                           _LAYER_PREFIX_RE.sub("s_.l_.", text)))
    if hashable_filtered:
        if len(_META_TEXT_CACHE) >= _META_CACHE_LIMIT:
            _META_TEXT_CACHE.clear()
        _META_TEXT_CACHE[filtered] = key
    if len(_META_ID_CACHE) >= _META_CACHE_LIMIT:
        _META_ID_CACHE.clear()
    _META_ID_CACHE[id(metadata)] = (metadata, key)
    return key


def _op_meta_key(op) -> Any:
    """Canonical occurrence-relative projection of one operator's metadata."""
    from .physical_key import operator_metadata_key
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


def _execution_geometry(device):
    """A representative shard requires identical physical owner topology."""
    roles = dict(device.rates).keys() | {"cube", "host"}
    return (
        device.compute_units,
        tuple(sorted((role, device.units(role)) for role in roles)),
        tuple(sorted((pool, device.pool_units(pool)) for pool, _ in device.buffers)),
        tuple(sorted((mode, role) for mode, (role, _) in device.service_modes)),
        tuple(sorted(dict(device.compute_peaks))),
        tuple(sorted((role, dict(device.work_units).get(role, NATIVE_WORK_UNITS.get(role, "work")))
                     for role in roles | dict(device.service_modes).keys())),
    )


@dataclass(frozen=True)
class StageProfile:
    """Immutable cluster-level rates, pools and die geometry for one stage."""

    replica: int
    stage: int
    dies: tuple
    geometry: tuple
    rates: dict
    names: dict
    resources: tuple
    stage_pools: dict
    pools: dict
    pool_units: dict
    pool_units_by_name: dict
    units: dict
    compute_peaks: dict
    pool_capacities: dict
    service_modes: dict
    work_units: dict
    store_queues: dict
    physical_mechanisms: Any

    @property
    def profile_key(self) -> tuple:
        return (int(self.stage), self.geometry)


def build_stage_profiles(case: CaseSpec, namespace: str = "", *,
                         replicas=None) -> tuple[StageProfile, ...]:
    """Resolve every stage's immutable hardware binding exactly once."""
    p = case.runtime.parallel
    width = p.tp * p.pcp
    selected = range(p.dp) if replicas is None else tuple(replicas)
    profiles: list[StageProfile] = []
    for replica in selected:
        for stage in range(p.pp):
            start = (int(replica) * p.pp + stage) * width
            dies = tuple(case.machine.dies[start:start + width])
            if len(dies) != width:
                raise ModelingError("placement_exceeds_physical_devices")
            geometry = _execution_geometry(dies[0])
            if any(_execution_geometry(die) != geometry for die in dies[1:]):
                raise ModelingError(
                    "representative_shard_requires_matching_resource_geometry:"
                    f"replica={replica}:stage={stage};"
                    "heterogeneous_owners_require_per_rank_execution")
            roles = dict(dies[0].rates)
            work_units = {**NATIVE_WORK_UNITS, **dict(dies[0].work_units)}
            rates = {
                role: (min(dict(die.rates)[role] for die in dies) / 1000,
                       work_units.get(role, "work"))
                for role in roles
            }
            rates.update({"cube": (1, "peak_ms"), "host": (1, "ms")})
            names: dict[str, str] = {}
            stage_resources: list[Resource] = []
            for kind, (rate, unit) in rates.items():
                label = {"cube": "matrix", "hbm": "memory"}.get(kind, kind)
                name = f"{namespace}{label}@r{replica}:s{stage}"
                stage_resources.append(Resource(
                    name, rate, unit, "fifo" if kind == "host" else "fair",
                    dies[0].units(kind)))
                names[kind] = name
            stage_pools: dict[str, str] = {}
            pools: dict[str, float] = {}
            pool_units: dict[str, int] = {}
            for pool in dict(dies[0].buffers):
                key = f"{namespace}{pool}@r{replica}:s{stage}"
                pools[key] = min(dict(die.buffers)[pool] for die in dies)
                pool_units[key] = dies[0].pool_units(pool)
                stage_pools[str(pool)] = key
            compute_peaks = {
                dtype: min(die.peak(dtype) for die in dies)
                for dtype, _ in dies[0].compute_peaks
                if dtype not in dict(dies[0].compute_peak_aliases)
            }
            service_modes = {
                mode: (role, min(dict(die.service_modes)[mode][1] for die in dies))
                for mode, (role, _) in dies[0].service_modes
            }
            store_queues = {
                name: dict(queue) for name, queue in dies[0].store_queues
                if all(dict(die.store_queues).get(name) == queue for die in dies)
            }
            profiles.append(StageProfile(
                replica=int(replica), stage=int(stage), dies=dies,
                geometry=geometry, rates=rates, names=names,
                resources=tuple(stage_resources), stage_pools=stage_pools,
                pools=pools, pool_units=pool_units,
                units={role: dies[0].units(role) for role in names},
                pool_units_by_name={pool: dies[0].pool_units(pool)
                                    for pool in dict(dies[0].buffers)},
                compute_peaks=compute_peaks,
                pool_capacities={pool: min(dict(die.buffers)[pool] for die in dies)
                                 for pool in dict(dies[0].buffers)},
                service_modes=service_modes, work_units=work_units,
                store_queues=store_queues,
                physical_mechanisms=dies[0].physical_mechanisms,
            ))
    return tuple(profiles)


@dataclass(frozen=True)
class SubmissionPolicy:
    graph_replay_ms: float | None = None
    scheduler_ms: float = 0
    completion_ms: float = 0
    kernel_launch_ms: float | None = None
    piecewise_sync_ms: float = 0.0

    def launch_ms(self, dies, *, captured=False):
        """Software submission overrides the explicit legacy device parameter."""
        eager = (self.kernel_launch_ms if self.kernel_launch_ms is not None
                 else max(d.launch_us for d in dies) / 1000)
        return (self.graph_replay_ms if captured and self.graph_replay_ms is not None
                else eager)


@dataclass
class ScheduledIteration:
    replica: int
    iteration: int
    completion: Event
    start_ms: float
    mode: str
    valid_tokens: int
    padded_tokens: int
    graph_bucket: int | None
    replay_scope: str
    operators: int
    commits: Mapping[str, Event]
    device_complete: Event
    host_results: Mapping[str, Event]
    client_deliveries: Mapping[str, Event]
    implementation_selections: tuple[dict, ...] = ()
    #: (replica, stage, actual host-launch start event) created by this step.
    host_submits: tuple[tuple[int, int, Event], ...] = ()
    #: Requests whose token_commit operator actually executed in this step.
    declared_commits: Mapping[str, Event] = field(default_factory=dict)
    #: Requests with no declared token_commit, completed by the step fallback.
    fallback_commits: Mapping[str, Event] = field(default_factory=dict)
    #: Actual last-use boundary after this step's owned copies; T14 may use it
    #: to wake a scheduler without reconstructing free/grant names.
    offload_release: Event | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica": self.replica, "iteration": self.iteration,
            "start_ms": self.start_ms, "end_ms": self.completion.time,
            "duration_ms": self.completion.time - self.start_ms,
            "mode": self.mode, "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens, "graph_bucket": self.graph_bucket,
            "replay_scope": self.replay_scope, "operators": self.operators,
            "device_end_ms": self.device_complete.time,
            "host_result_ready_ms": max((event.time for event in self.host_results.values()), default=None),
            "implementation_selections": list(self.implementation_selections),
            "host_submits": [
                {"replica": replica, "stage": stage,
                 "start_ms": event.time}
                for replica, stage, event in self.host_submits],
            "declared_commit_ms": {
                request: event.time
                for request, event in self.declared_commits.items()},
            "fallback_commit_ms": {
                request: event.time
                for request, event in self.fallback_commits.items()},
            "offload_release_ms": (
                None if self.offload_release is None
                else self.offload_release.time),
        }


@dataclass(frozen=True)
class CollectiveParticipant:
    replica: int
    ready: Event
    context: RecipeContext
    recipe: Any
    name: str
    completion: Event


class _RankEngine:
    """Name-mapping view over the one shared Engine.

    It does not schedule, allocate a second clock, or own tasks.  It only
    translates a launch's native pipe names (including hardcoded source keys
    such as ``pipe_s_issue``) to that rank's physical resources before the
    shared Engine prepares the task.
    """

    def __init__(self, base, resource_names):
        self.base = base
        self.resource_names = {str(key): str(value)
                               for key, value in dict(resource_names).items()}

    def _work(self, work):
        return {self.resource_names.get(str(key), str(key)): value
                for key, value in dict(work or {}).items()}

    def _keys(self, values):
        return {self.resource_names.get(str(key), str(key)): value
                for key, value in dict(values or {}).items()}

    def submit(self, name, work=None, **kwargs):
        return self.base.submit(
            name, self._work(work) if work is not None else None, **kwargs)

    def _prepare(self, name, work=None, *, latency=0.0, caps=None,
                 acquire=(), free=(), placement=None, coupled=False,
                 priority=0):
        return self.base._prepare(
            name, self._work(work) if work is not None else None,
            latency=latency, caps=self._keys(caps), acquire=acquire,
            free=free, placement=self._keys(placement), coupled=coupled,
            priority=priority)

    def _submit_prepared(self, prepared, name, **kwargs):
        return self.base._submit_prepared(prepared, name, **kwargs)

    def future(self, name):
        return self.base.future(name)

    def resolve(self, event):
        return self.base.resolve(event)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _bind_native_resource_view(program, resource_names, engine):
    """Shallow launch-local resource view; cached program is not mutated."""
    if not resource_names:
        return program
    from .program import NativeProgram
    inner = copy(program.program)
    names = dict(getattr(inner, "names", {}) or {})
    rates = dict(getattr(inner, "rates", {}) or {})
    units = dict(getattr(inner, "units", {}) or {})
    for pipe, physical in dict(resource_names).items():
        names[str(pipe)] = str(physical)
        resource = engine.resources.get(str(physical))
        if resource is not None:
            rates[str(pipe)] = resource.rate
            rates[str(physical)] = resource.rate
            units[str(pipe)] = resource.units
            units[str(physical)] = resource.units
    inner.names = names
    inner.rates = rates
    inner.units = units
    return NativeProgram(inner, program.work)


def launch_native_cohort(engine, launches, *, name_prefix="native_cohort",
                         owner="native_cohort", defer=False):
    """Two-phase launch of native peer programs on one shared Engine.

    Each launch receives a private launch-local rank/topology view; the cached
    native program is never mutated.  Directed ``(phase, src_rank, dst_rank)``
    futures are allocated before any program starts.  After every program has
    produced its real per-phase send handles, each future is bound to the
    matching sender's actual send Event.  Receivers wait the future, so only the
    named peer/phase join is consumed.  Per-rank parent Events depend only on
    that rank's own program finish; the cohort join waits every real finish.

    ``launches`` is a sequence of mappings (or attribute objects) with:
      program: NativeProgram
      name: unique launch name
      ready: actual local ready Event
      rank_index: launch-local EP rank index
      peer_count / peer_labels / node_ranks / peer_classes / peer_weights:
        optional launch-local geometry.  Absent fields stay the program's own
        declared values; no scalar wait or measured timestamp is accepted.

    Returns finishes, per-rank parents, the genuine global join, launch-local
    handles, and unresolved peer labels.
    """
    from .program import NativeProgram
    from .mc2_program import Mc2LaunchView

    def field(raw, key, default=None):
        return (raw.get(key, default) if isinstance(raw, _AbcMapping)
                else getattr(raw, key, default))

    infos, by_rank, names = [], {}, set()
    phases = None
    for raw in launches:
        program = field(raw, "program")
        if not isinstance(program, NativeProgram):
            raise ModelingError("native_cohort_requires_native_program")
        name = str(field(raw, "name") or "")
        if not name or name in names:
            raise ModelingError(f"native_cohort_launch_name_invalid:{name}")
        names.add(name)
        rank = field(raw, "rank_index")
        if rank is None:
            raise ModelingError(f"native_cohort_rank_missing:{name}")
        rank = int(rank)
        if rank in by_rank:
            raise ModelingError(f"native_cohort_rank_duplicate:{rank}")
        inner = program.program
        source_view = getattr(inner, "view", None)
        peer_count = int(field(raw, "peer_count")
                         or getattr(inner, "peer_count", 0) or 0)
        if peer_count <= 0:
            raise ModelingError(f"native_cohort_peer_count_missing:{name}")
        if not 0 <= rank < peer_count:
            raise ModelingError(f"native_cohort_rank_out_of_range:{name}:{rank}")
        labels = field(raw, "peer_labels")
        labels = (tuple(int(item) for item in labels)
                  if labels is not None else
                  tuple(item for item in range(peer_count) if item != rank))
        if len(set(labels)) != len(labels) or (
                rank in labels
                or any(not 0 <= item < peer_count for item in labels)):
            raise ModelingError(f"native_cohort_peer_labels_invalid:{name}")
        node_ranks = field(raw, "node_ranks")
        node = int(node_ranks if node_ranks is not None else
                   (getattr(source_view, "node_ranks", 0)
                    if source_view is not None else 0))
        peer_classes = field(raw, "peer_classes")
        classes = tuple(peer_classes or ())
        if not classes and source_view is not None and source_view.rank == rank:
            classes = tuple(source_view.peer_classes)
        peer_weights = field(raw, "peer_weights")
        weights = tuple(peer_weights or ())
        if not weights and source_view is not None and source_view.rank == rank:
            weights = tuple(source_view.peer_weights)
        intra = field(raw, "intra_latency_ms")
        inter = field(raw, "inter_latency_ms")
        resource_names = field(raw, "resource_names")
        frames_raw = field(raw, "payload_frames", None)
        if frames_raw is None:
            # An exact source ledger is indexed by physical source rank, not
            # by the representative graph's rank. The same program is launched
            # once per replica with its own whole routed frames.
            frames_for = getattr(inner, "frames_for", None)
            if callable(frames_for):
                frames_raw = frames_for(rank)
        if frames_raw is None and source_view is not None:
            frames_raw = getattr(source_view, "payload_frames", None)
        if isinstance(frames_raw, _AbcMapping):
            payload_frames = tuple(sorted(
                (int(peer), int(frames)) for peer, frames in frames_raw.items()
                if int(frames) > 0))
        elif frames_raw:
            payload_frames = tuple(sorted(
                (int(item[0]), int(item[1])) for item in frames_raw
                if int(item[1]) > 0))
        else:
            payload_frames = ()
        packet = field(raw, "payload_packet_bytes", None)
        if packet is None and source_view is not None:
            packet = getattr(source_view, "payload_packet_bytes", 0)
        payload_packet = int(packet or 0)
        basis_raw = field(raw, "payload_basis", None)
        if basis_raw is None:
            basis_for = getattr(inner, "payload_basis_for", None)
            if callable(basis_for):
                basis_raw = basis_for(rank)
        if basis_raw is None:
            basis_raw = (getattr(source_view, "payload_basis", "unknown")
                         if source_view is not None else "unknown")
        payload_basis = str(basis_raw)
        local_frames = field(raw, "payload_local_frames", None)
        if local_frames is None:
            local_for = getattr(inner, "local_frames_for", None)
            if callable(local_for):
                local_frames = local_for(rank)
        if local_frames is None and source_view is not None:
            local_frames = getattr(source_view, "payload_local_frames", 0)
        payload_local = int(local_frames or 0)
        view = Mc2LaunchView(
            rank=rank, peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=node,
            intra_latency_ms=(getattr(source_view, "intra_latency_ms", 0)
                              if intra is None and source_view is not None
                              else (intra or 0)),
            inter_latency_ms=(getattr(source_view, "inter_latency_ms", 0)
                              if inter is None and source_view is not None
                              else (inter or 0)),
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(
                    resource_names
                    or (source_view.resource_names if source_view is not None
                        else ())).items())),
            payload_frames=payload_frames,
            payload_packet_bytes=payload_packet,
            payload_basis=payload_basis,
            payload_local_frames=payload_local)
        launch_phases = tuple(getattr(inner, "phases", ()))
        phase_names = tuple(getattr(phase, "name", str(phase))
                            for phase in launch_phases)
        if phases is None:
            phases = phase_names
        elif set(phases) != set(phase_names):
            raise ModelingError(
                f"native_cohort_phase_mismatch:{name}:"
                f"{sorted(set(phases) ^ set(phase_names))}")
        infos.append({
            "name": name, "rank": rank, "view": view,
            "program": program, "ready": field(raw, "ready"),
            "operator_id": str(field(raw, "operator_id", "mc2")),
            "resource_names": dict(resource_names or {}),
            "payload_frames": dict(payload_frames),
            "payload_packet_bytes": payload_packet,
            "payload_basis": payload_basis,
            "payload_local_frames": payload_local,
            "transport": field(raw, "transport", None),
        })
        by_rank[rank] = infos[-1]
    if not infos:
        raise ModelingError("native_cohort_empty")
    link_phases = tuple(
        phase for phase in (phases or ())
        if any(getattr(info["program"].program, "link_by_phase", {}).get(phase)
               for info in infos))
    for info in infos:
        info["program"] = _bind_native_resource_view(
            info["program"], info["resource_names"], engine)
    programs_by_rank = {info["rank"]: info["program"] for info in infos}
    names_by_rank = {info["rank"]: info["name"] for info in infos}
    unresolved, partial = [], []
    placeholders = {}
    # Expected directed cells are real local receive prerequisites even when
    # no witnessed integer frame count exists.  Prebind the same futures for
    # every cell a launch program will resolve; receivers wait these handles
    # instead of a global phase barrier.
    for info in infos:
        transport = info.get("transport")
        inner = info["program"].program
        if transport is None or not hasattr(inner, "transport_cells"):
            continue
        for phase_name, count_phase in (
                ("payload_exchange", False), ("count_status_exchange", True)):
            if phase_name not in (phases or ()):
                continue
            for _si, di, _src, _dst, useful, _frames in inner.transport_cells(
                    transport, count=count_phase):
                if useful <= 0 or _si == di:
                    continue
                placeholders.setdefault(
                    (_si, di, phase_name),
                    engine.future(f"{name_prefix}:{phase_name}:{_si}->{di}"))
    protocol_edges = 0
    directed_edges = 0
    directed_frames = 0
    fullmesh_by_rank = {}
    payload_packet_by_rank = {}
    for info in infos:
        source_inner = info["program"].program
        view = info["view"]
        rank = int(info["rank"])
        source_phases = tuple(
            (str(phase), links) for phase, links
            in getattr(source_inner, "link_by_phase", {}).items() if links)
        fullmesh = tuple(sorted(
            phase for phase, _links in source_phases
            if phase != "payload_exchange"))
        fullmesh_by_rank[info["name"]] = fullmesh
        for phase, _links in source_phases:
            if phase != "payload_exchange":
                # Count/control is a real fullmesh protocol: every expected
                # peer participates.
                for peer in view.peer_labels:
                    if peer in programs_by_rank:
                        key = (rank, peer, phase)
                        if key not in placeholders:
                            placeholders[key] = engine.future(
                                f"{name_prefix}:{phase}:{rank}->{peer}")
                        protocol_edges += 1
                    else:
                        edge = (phase, rank, peer)
                        if edge not in unresolved:
                            unresolved.append(edge)
                continue
            frames = dict(info.get("payload_frames") or {})
            basis = str(info.get("payload_basis") or "unknown")
            packet = int(info.get("payload_packet_bytes") or 0)
            payload_packet_by_rank[info["name"]] = packet
            if frames:
                if packet <= 0:
                    partial.append({
                        "rank": rank, "name": info["name"],
                        "phase": "payload_exchange", "basis": basis,
                        "reason": "payload_packet_bytes_missing"})
                    continue
                for peer, whole_frames in frames.items():
                    peer = int(peer)
                    whole_frames = int(whole_frames)
                    if whole_frames <= 0:
                        continue
                    if peer not in view.peer_labels:
                        partial.append({
                            "rank": rank, "name": info["name"],
                            "phase": "payload_exchange", "peer": peer,
                            "basis": basis,
                            "reason": "destination_not_declared"})
                        continue
                    if peer in programs_by_rank:
                        key = (rank, peer, "payload_exchange")
                        if key not in placeholders:
                            placeholders[key] = engine.future(
                                f"{name_prefix}:payload_exchange:"
                                f"{rank}->{peer}")
                        directed_edges += 1
                        directed_frames += whole_frames
                    else:
                        edge = ("payload_exchange", rank, peer)
                        if edge not in unresolved:
                            unresolved.append(edge)
                continue
            if "zero" in basis.lower():
                # Explicit known zero: no future, no invented arrival.
                continue
            partial.append({
                "rank": rank, "name": info["name"],
                "phase": "payload_exchange", "basis": basis,
                "reason": "payload_destinations_unknown"})
        if not any(phase == "payload_exchange"
                   for phase, _links in source_phases):
            frames = dict(info.get("payload_frames") or {})
            basis = str(info.get("payload_basis") or "unknown")
            if not frames and "zero" not in basis.lower():
                partial.append({
                    "rank": rank, "name": info["name"],
                    "phase": "payload_exchange", "basis": basis,
                    "reason": "payload_destinations_unknown"})
    def _declared_metrics(info):
        inner = info["program"].program
        transport = info.get("transport") or {}
        rank = int(info["rank"])
        payload, count = (inner.transport_account(transport)
                          if hasattr(inner, "transport_account") else ({}, {}))
        phases = {phase.name for phase in getattr(inner, "phases", ())}
        records = {}
        if payload:
            records["payload_exchange"] = {
                **payload,
                "basis": (inner.payload_basis_for(rank)
                          if hasattr(inner, "payload_basis_for")
                          else info["payload_basis"]),
                "count": False,
            }
        if count and "count_status_exchange" in phases:
            count_basis = (inner.payload_basis_for(rank)
                           if hasattr(inner, "payload_basis_for")
                           else info["payload_basis"])
            records["count_status_exchange"] = {
                **count,
                "basis": count_basis,
                "count": True,
            }
        links_for_rank = getattr(inner, "link_by_phase_for", None)
        links = (dict(links_for_rank(rank)) if callable(links_for_rank)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        work_for_rank = getattr(inner, "work_for_rank", None)
        work = (tuple(work_for_rank(rank)) if callable(work_for_rank)
                else tuple(getattr(info["program"], "work", ()) or ()))
        transport_work = dict(payload.get("work", {}))
        if "count_status_exchange" in phases:
            for pipe, amount in (count.get("work", {}) or {}).items():
                transport_work[pipe] = transport_work.get(pipe, 0.0) + float(amount)
        return records, links, work, transport_work

    cohort = {}
    if defer:
        launch_by_rank = {info["rank"]: info for info in infos}
        parents, starts = {}, {}
        remaining_starts = [len(infos)]
        for info in infos:
            name = info["name"]
            parent = engine.future(name + ":program_complete")
            parents[name] = parent
            starts[name] = info["ready"]
            records, links, work, transport_work = _declared_metrics(info)
            cohort[info["rank"]] = {
                "parent": parent, "start": info["ready"],
                "operators": {}, "phases": {},
                "handles": {"deferred": True},
                "work": work,
                "link_by_phase": {
                    phase: {str(pipe): float(amount)
                            for pipe, amount in values.items()}
                    for phase, values in links.items()},
                "transport": records,
                "transport_work": transport_work,
            }

            def begin(_event, item=info, promise=parent):
                rank_engine = _RankEngine(engine, item["resource_names"])
                finish, _handles = item["program"].start_with_handles(
                    rank_engine, item["name"], item["ready"],
                    launch=item["view"], send_placeholders=placeholders,
                    transport=item.get("transport"))
                engine.submit(item["name"] + ":parent_finish", after=(finish,),
                              owner=owner,
                              on_complete=lambda _e: engine.resolve(promise))
                remaining_starts[0] -= 1
                if remaining_starts[0] == 0:
                    # The handles are now owned by the real receive/retire
                    # events; do not retain the per-cohort sentinel registry.
                    placeholders.clear()

            engine.submit(name + ":begin", after=(info["ready"],), owner=owner,
                          on_complete=begin)
        parents = {names_by_rank[rank]: item["parent"]
                   for rank, item in cohort.items()}
        starts = {names_by_rank[rank]: item["start"]
                  for rank, item in cohort.items()}
        operators = {names_by_rank[rank]: item["operators"]
                     for rank, item in cohort.items()}
        handles = {names_by_rank[rank]: item["handles"]
                   for rank, item in cohort.items()}
        phases_by_rank = {names_by_rank[rank]: item["phases"]
                          for rank, item in cohort.items()}
        work_by_rank = {names_by_rank[rank]: item["work"]
                        for rank, item in cohort.items()}
        link_by_phase_by_rank = {
            names_by_rank[rank]: item["link_by_phase"]
            for rank, item in cohort.items()}
        transport_by_rank = {
            names_by_rank[rank]: item["transport"]
            for rank, item in cohort.items()}
        transport_work_by_rank = {
            names_by_rank[rank]: item["transport_work"]
            for rank, item in cohort.items()}
        transport_tx = sum(
            float(phase.get("tx_bytes", 0.0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_rx = sum(
            float(phase.get("rx_bytes", 0.0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_frames = sum(
            int(phase.get("frames", 0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_edges = sum(
            int(phase.get("edges", 0))
            for details in transport_by_rank.values()
            for phase in details.values())
        join = engine.submit(f"{name_prefix}:join", after=tuple(parents.values()),
                             owner=owner)
        return {
            "schema": "mindie-sim/native-cohort/3",
            "finishes": parents, "parents": parents, "join": join,
            "starts": starts, "operators": operators, "handles": handles,
            "phases_by_rank": phases_by_rank, "work_by_rank": work_by_rank,
            "link_by_phase_by_rank": link_by_phase_by_rank,
            "transport_by_rank": transport_by_rank,
            "transport_work_by_rank": transport_work_by_rank,
            "transport_tx_bytes": transport_tx,
            "transport_rx_bytes": transport_rx,
            "transport_frames": transport_frames,
            "transport_edges": transport_edges,
            "directed_edges": sum(
                int(value.get("edges", 0))
                for details in transport_by_rank.values()
                for phase, value in details.items()
                if isinstance(value, _AbcMapping) and not value.get("count")),
            "directed_frames": transport_frames,
            "rank_indices": {names_by_rank[rank]: rank for rank in cohort},
            "payload_basis": {
                names_by_rank[rank]: str(
                    getattr(launch_by_rank[rank]["program"].program,
                            "payload_basis_for",
                            lambda _r, info=launch_by_rank[rank]:
                            info["payload_basis"])(rank))
                for rank in cohort},
            "payload_packet_bytes": {
                names_by_rank[rank]: int(
                    getattr(launch_by_rank[rank]["program"].program, "view", None).payload_packet_bytes
                    if getattr(launch_by_rank[rank]["program"].program, "view", None) is not None else 0)
                for rank in cohort},
            "payload_frames": {
                names_by_rank[rank]: dict(
                    getattr(launch_by_rank[rank]["program"].program,
                            "frames_for", lambda _r: ())(rank))
                for rank in cohort},
            "protocol_edges": sum(
                1 for key in placeholders
                if key[2] not in {"payload_exchange"}),
            "unresolved_edges": len(unresolved),
            "partial": tuple(partial),
        }
    for info in infos:
        rank_engine = _RankEngine(engine, info["resource_names"])
        parent, launch_handles = info["program"].start_with_handles(
            rank_engine, info["name"], info["ready"],
            launch=info["view"], send_placeholders=placeholders,
            transport=info.get("transport"))
        op_id = info.get("operator_id", "mc2")
        operators = dict(launch_handles.get("operators") or {})
        if op_id != "mc2" and "mc2" in operators:
            operators = {op_id: operators["mc2"]}
        inner = getattr(info["program"], "program", None)
        work_for_rank = getattr(inner, "work_for_rank", None)
        work = (tuple(work_for_rank(info["rank"]))
                if callable(work_for_rank)
                else tuple(getattr(info["program"], "work", ()) or ()))
        links_for_rank = getattr(inner, "link_by_phase_for", None)
        links = (dict(links_for_rank(info["rank"]))
                 if callable(links_for_rank)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        cohort[info["rank"]] = {
            "parent": parent,
            "start": launch_handles.get("start", info["ready"]),
            "operators": operators,
            "phases": dict(launch_handles.get("phase_complete") or {}),
            "handles": launch_handles,
            "work": work,
            "link_by_phase": {
                phase: {str(pipe): float(amount) for pipe, amount in values.items()}
                for phase, values in links.items()},
            "transport": dict(launch_handles.get("transport") or {}),
            "transport_work": dict(launch_handles.get("transport_work") or {}),
        }
    parents = {names_by_rank[rank]: item["parent"]
               for rank, item in cohort.items()}
    starts = {names_by_rank[rank]: item["start"]
              for rank, item in cohort.items()}
    operators = {names_by_rank[rank]: item["operators"]
                 for rank, item in cohort.items()}
    handles = {names_by_rank[rank]: item["handles"]
               for rank, item in cohort.items()}
    phases_by_rank = {names_by_rank[rank]: item["phases"]
                      for rank, item in cohort.items()}
    work_by_rank = {names_by_rank[rank]: item["work"]
                    for rank, item in cohort.items()}
    link_by_phase_by_rank = {
        names_by_rank[rank]: item["link_by_phase"]
        for rank, item in cohort.items()}
    transport_by_rank = {
        names_by_rank[rank]: item["transport"]
        for rank, item in cohort.items()}
    transport_work_by_rank = {
        names_by_rank[rank]: item["transport_work"]
        for rank, item in cohort.items()}
    transport_tx = sum(
        float(phase.get("tx_bytes", 0.0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_rx = sum(
        float(phase.get("rx_bytes", 0.0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_frames = sum(
        int(phase.get("frames", 0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_edges = sum(
        int(phase.get("edges", 0))
        for details in transport_by_rank.values()
        for phase in details.values())
    join = engine.submit(f"{name_prefix}:join", after=tuple(parents.values()),
                         owner=owner)
    return {
        "schema": "mindie-sim/native-cohort/3",
        "finishes": parents, "parents": parents, "join": join,
        "starts": starts, "operators": operators, "handles": handles,
        "phases_by_rank": phases_by_rank, "work_by_rank": work_by_rank,
        "link_by_phase_by_rank": link_by_phase_by_rank,
        "transport_by_rank": transport_by_rank,
        "transport_work_by_rank": transport_work_by_rank,
        "transport_tx_bytes": transport_tx,
        "transport_rx_bytes": transport_rx,
        "transport_frames": transport_frames,
        "transport_edges": transport_edges,
        "by_rank": {rank: dict(item) for rank, item in cohort.items()},
        "names_by_rank": dict(names_by_rank),
        "phases": tuple(phases or ()), "link_phases": link_phases,
        "fullmesh_phases": fullmesh_by_rank,
        "payload_frames": {
            info["name"]: dict(info.get("payload_frames") or {})
            for info in infos},
        "payload_basis": {
            info["name"]: str(info.get("payload_basis") or "unknown")
            for info in infos},
        "payload_packet_bytes": payload_packet_by_rank,
        "partial": tuple(partial),
        "protocol_edges": protocol_edges,
        "directed_edges": directed_edges,
        "directed_frames": directed_frames,
        "unresolved_peers": tuple(sorted({peer for _, _, peer in unresolved})),
        "unresolved_edges": len(unresolved),
        "rank_indices": {info["name"]: info["rank"] for info in infos},
    }


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
    """CostModel view whose diagnostic records belong to one execution.

    Pricing logic and immutable hardware/service declarations stay on the
    shared model; every mutable counter, mapping, invocation and regime record
    goes to this execution's private ``CostLedger``.  This is the read/write
    boundary that keeps an isolated row's ``basis`` isolated while preserving
    the historical ``execution.cost`` public surface.
    """

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
    """One selected collective algorithm and its lazily built topology plan.

    ``algorithm`` lets a recipe consume the exact plan execution selected
    instead of re-deriving it; a prebuilt plan keeps the public ledger on the
    same physical messages.
    """

    algorithm: str
    plan: object | None = None
    factory: object | None = None

    def __call__(self):
        if self.plan is not None:
            return self.plan
        if self.factory is None:
            raise ModelingError("collective_plan_factory_missing")
        return self.factory()


#: Bounded live-frontier ceiling above which eager admission defers to the
#: lazily expanded ``:begin`` boundary.
INLINE_FRONTIER_TASK_CAP = 49152
#: Per-Execution lowered-program cache bound (occurrence-keyed, not results).
PROGRAM_CACHE_ENTRY_LIMIT = 768

class Execution:
    # Identical private schedules replay across ranks and later layers.
    # Experimental: off in the shipped default. Opt-in only; no shape or input auto-enable.
    schedule_replay = False
    _isolated_scripts: dict = {}

    def __init__(
        self, case: CaseSpec, cost: CostModel, *, recipes: RecipeBook | None = None,
        submission: SubmissionPolicy | None = None, trace_limit: int = ZERO,
        engine: Engine | None = None, namespace: str = "",
        include_submission: bool = True, on_operator=None,
        include_serving_io: bool = False,
        client_tails=None, memory_policy=None,
        legacy_diagnostics: bool | None = None,
        fragment_state=None, memory=None,
        lazy_frontier: bool = True, host_lookahead: int = HOST_LOOKAHEAD_DEFAULT,
        bindings=None, ledger=None, isolated_replica: int | None = None,
        evaluation=None, accel_eager_attach: bool = False,
    ):
        from .cost import CostLedger
        self.case, self.cost = case, cost
        self.bindings = bindings
        # An explicitly passed evaluation context owns the bounded prepared
        # program cache shared by graph, cohort and operator pricing.  It is
        # never a module-global and never outlives one public evaluation.
        self.evaluation = evaluation
        self._shared_eval_cache = (
            None if evaluation is None else evaluation.prepared_programs)
        # Mutable diagnostic records belong to this execution, not to the
        # shared immutable pricing model.
        self.ledger = ledger if ledger is not None else CostLedger()
        self.cost_model = cost
        self.cost = _LedgerBoundCost(cost, self.ledger)
        self.namespace = namespace
        self.memory = memory
        if legacy_diagnostics is None:
            metadata = getattr(case.runtime, "metadata", {}) or {}
            if not isinstance(metadata, _AbcMapping):
                metadata = dict(metadata)
            # An explicit PrimitiveStore is a historical diagnostic source.
            # Ordinary production passes no primitive store and uses declared
            # physical work + normalized resource/phase services.
            legacy_diagnostics = bool(
                metadata.get("legacy_diagnostics", False)
                or getattr(cost, "primitives", None) is not None)
        # Ordinary production uses physical work + normalized resource/phase
        # services. The old measured-primitive completion floor and observed
        # MAC-as-work path remain available only when explicitly requested for
        # historical diagnostics.
        self.legacy_diagnostics = bool(legacy_diagnostics)
        self.include_submission = include_submission
        self.on_operator = on_operator
        self.lazy_frontier = bool(lazy_frontier)
        self.host_lookahead = max(1, int(host_lookahead))
        self.accel_eager_attach = bool(accel_eager_attach)
        self.recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        from .invocation_binding import InvocationPolicy
        self.invocations = InvocationPolicy(case.runtime.kernels.get("invocations", ()))
        from .service_regimes import RegimePolicy
        self.regimes = RegimePolicy(case.runtime.kernels.get("service_regimes", ()))
        self.submission = submission or SubmissionPolicy(**case.runtime.submission)
        self.submission_source = "api.submission" if submission is not None else "runtime.submission"
        self.serving_io = ServingIO(case.runtime.serving_io if include_serving_io else {},
                                    namespace, case.runtime.parallel.dp, client_tails)
        from .offload import offload_resources, resolve_case_offload_policy
        self.offload_policy = resolve_case_offload_policy(case, memory_policy)
        self._offload_memory = None
        self.step_summary = StepSummary()
        self.stage_names: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_pools: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_dies: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.streams: dict[tuple[int, int, str], Event] = {}
        self.hosts: dict[tuple[int, int], Event] = {}
        self.submission_counts = {"eager_invocations": 0, "graph_replays": 0}
        self.prepared = {}
        self.schedules = {}
        resources, pools, pool_units = [], {}, {}
        resources.extend(self.serving_io.resources())
        resources.extend(offload_resources(
            self.offload_policy, namespace, case.runtime.parallel.dp, case.runtime.parallel.pp))
        p = case.runtime.parallel
        selected_replica = (
            None if isolated_replica is None else int(isolated_replica))
        self.stage_profiles: dict[tuple[int, int], StageProfile] = {}
        profiles = None
        if (bindings is not None and not namespace
                and getattr(bindings, "stage_profiles", ())):
            profiles = tuple(
                profile for profile in bindings.stage_profiles
                if selected_replica is None or profile.replica == selected_replica)
        if not profiles:
            profiles = build_stage_profiles(
                case, namespace,
                replicas=(None if selected_replica is None
                          else (selected_replica,)))
        if selected_replica is not None and not profiles:
            raise ModelingError("isolated_replica_out_of_range")
        active_ranks: set[int] = set()
        scheduler_replicas: set[int] = set()
        for profile in profiles:
            replica, stage = profile.replica, profile.stage
            if replica not in scheduler_replicas:
                resources.append(Resource(
                    f"{namespace}scheduler@r{replica}", 1, "ms", "fifo"))
                scheduler_replicas.add(replica)
            self.stage_dies[replica, stage] = profile.dies
            self.stage_names[replica, stage] = profile.names
            self.stage_pools[replica, stage] = profile.stage_pools
            self.stage_profiles[replica, stage] = profile
            resources.extend(profile.resources)
            pools.update(profile.pools)
            pool_units.update(profile.pool_units)
            active_ranks.update(int(die.rank) for die in profile.dies)
        self.topology = (
            bindings.topology
            if bindings is not None and bindings.topology is not None
            and not namespace
            else Topology(case.machine, p, namespace,
                          collective_startup_mode=case.runtime.collective_startup_mode))
        if selected_replica is None:
            resources.extend(self.topology.resources())
        else:
            resources.extend(self.topology.resources_for_ranks(active_ranks))
        if engine is None and self.memory is not None:
            # A Memory ledger already adopted by one RuntimeState owns that
            # single causal root.  Re-using the ledger therefore means re-using
            # its root: build no second Engine and raise no conflicting-clock
            # error.  An explicit ``engine=`` still wins (that caller declares
            # the binding itself).
            bound = getattr(self.memory, "runtime_state", None)
            if isinstance(bound, Engine):
                engine = bound
        self._engine_resource_templates = tuple(resources)
        self._engine_pool_templates = dict(pools)
        self._engine_pool_unit_templates = dict(pool_units)
        self._engine_trace_limit = int(trace_limit)
        if engine is not None:
            engine.extend(resources, pools, pool_units)
        self.engine = engine if engine is not None else Engine(
            resources, pools=pools, pool_units=pool_units, trace_limit=trace_limit,
            accel_eager_attach=self.accel_eager_attach)
        if self.memory is not None:
            if not self.engine.bind_memory(self.memory, pools=("hbm",)):
                raise ModelingError("execution_memory_ledger_conflict")
        self.resource_rates = ResourceRates(self.engine.resources)
        # Hardware bindings are stage properties. Thousands of pending kernels
        # retain their own work/readiness, but share these immutable inputs.
        self.stage_geometry = {key: Geometry.from_devices(dies)
                               for key, dies in self.stage_dies.items()}
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self.work = dict.fromkeys(("flops", "vector_ops", "hbm_bytes", "link_bytes"), 0.0)
        self.conflicts = 0
        self.recipes_used: dict[str, int] = {}
        self._collectives = None
        self._cohort_views = None
        self.native_group_joins: dict = {}
        # Bounded launch-local evidence: ranks, actual frames/edges, the same
        # per-rank link ledger and the work really submitted by the program.
        self.native_group_launches: dict = {}
        self._program_cache: dict = {}
        self._prepare_cache: dict = {}
        self._lazy_op_cache: dict = {}
        self._lazy_op_id_cache: dict = {}
        self._lazy_stage_id_caches: dict = {}
        self._lazy_safe_id_cache: dict = {}
        self._prev_lazy_dag = None
        self._prev_lazy_entries = None
        self._prev_add_topology = None
        self._compact_step: bool = False
        self._last_invocation_record = None
        self._service_identity_cache: dict = {}
        # Per-execution occurrence -> canonical metadata projection.  The
        # identity guard makes this a pure memo of one immutable declaration,
        # not a cross-case prediction cache.
        self._op_key_cache: dict = {}
        self._transport_options: dict = {}
        self.replay_counts = {"record": 0, "shadow": 0, "replay": 0, "real": 0}
        self._trace_key = None
        self._trace_shadows: list = []
        metadata = getattr(case.runtime, "metadata", {}) or {}
        if not isinstance(metadata, _AbcMapping):
            metadata = dict(metadata)
        self.readiness_report = metadata.get("collective_readiness")
        self.readiness_observations: list = []
        self.readiness_unresolved: list = []
        # Joint-collective observation is an opt-in bounded trace.  It retains
        # the first occurrence of each distinct joint collective, not per-step
        # history, and copies numeric boundaries rather than live Event handles.
        self._joint_collective_limit = max(0, int(trace_limit))
        self._joint_collective_trace = (
            {} if self._joint_collective_limit else None)
        self.joint_collective_stats = ({"observations": 0, "retained": 0,
                                        "max_ready_spread_ms": 0.0,
                                        "max_rendezvous_delay_ms": 0.0,
                                        "critical_rank": None}
                                       if self._joint_collective_trace is not None
                                       else None)
        self._joint_collective_ordinals: dict = {}
        from .participant_profile import normalise as _normalise_profile
        self.participant_profile = _normalise_profile(
            metadata.get("collective_participant_profile"))
        # Fragment assembly uses the same Engine and stage/pool bindings. The
        # conductor is kept per replica/stage only when a caller asks for the
        # convenient submit_fragment path; fragment_conductor can also be held
        # directly by an assembly/serving loop.
        self.fragment_conductors: dict = {}
        # One authoritative state ledger per RuntimeState. An explicitly
        # supplied StateLedger is adopted (and ownership-checked); a mapping
        # only seeds initial versions. A ledger owned by another runtime is
        # rejected instead of becoming a second truth.
        self.engine.ensure_state_ledger(fragment_state)
        # Optional SERVICE r2 actual access bindings. None means the offload
        # helper reads only actual state bindings in the current work; T14 can
        # set these from selected op/output events without a second runtime.
        self.state_accesses = None
        self.scheduled_accesses = None
        # Optional T06 normalized resource/phase services. They are consumed
        # only through an explicit op.metadata["resource_service"] binding; no
        # profile is guessed and no whole-call duration is used as a floor.
        from mindie_sim.observations.resource_services import normalize_resource_service
        self.resource_services = {}
        profile = metadata.get("resource_service_profile") if isinstance(
            metadata, Mapping) else None
        for record in (dict(profile or {}).get("services", ())
                       if isinstance(profile, _AbcMapping) else ()):
            service = normalize_resource_service(record)
            self.resource_services[service.service_id] = service

    def reset_engine(self) -> None:
        """Renew only the mutable Engine root for the next isolated price."""
        from .cost import CostLedger
        self.ledger = CostLedger()
        self.cost = _LedgerBoundCost(self.cost_model, self.ledger)
        self.recipes_used = {}
        self.engine = Engine(
            list(self._engine_resource_templates),
            pools=dict(self._engine_pool_templates),
            pool_units=dict(self._engine_pool_unit_templates),
            trace_limit=self._engine_trace_limit,
            accel_eager_attach=self.accel_eager_attach)
        if self.memory is not None:
            if not self.engine.bind_memory(self.memory, pools=("hbm",)):
                raise ModelingError("execution_memory_ledger_conflict")
        self.engine.ensure_state_ledger(None)
        self.resource_rates = ResourceRates(self.engine.resources)
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self._collectives = None
        self._cohort_views = None
        self.native_group_joins = {}
        self.native_group_launches = {}
        self._service_identity_cache = {}
        self._prev_lazy_dag = None
        self._prev_lazy_entries = None
        self._prev_add_topology = None

    def _op_key(self, op):
        key = self._op_key_cache.get(id(op))
        if key is not None and key[FIRST] is op:
            return key[SECOND]
        from .physical_key import operator_metadata_key
        value = operator_metadata_key(op.metadata, op.stage, op.layer)
        if len(self._op_key_cache) >= _META_CACHE_LIMIT:
            self._op_key_cache.clear()
        self._op_key_cache[id(op)] = (op, value)
        return value

    def ledger_recorder(self, method):
        """Bind one CostModel recorder to this execution's private ledger."""
        return _ledger_recorder(method, self.ledger)

    @property
    def fragment_state(self):
        """Live view of the one typed state ledger owned by this runtime."""
        return self.engine.ensure_state_ledger()

    def submit_catalog(self, catalog, binding, *, name: str, after=(),
                       owner: str = ""):
        """Submit one immutable catalog on this Execution's RuntimeState.

        This is the single public catalog entry for every consumer: the
        catalog and fresh ``BindingFrame`` go to the one Engine, so the
        returned ``Submission`` events, state versions, grants and receipts
        are all projections of the same mutable root. No second runtime, no
        duration reconstruction and no private event graph is created here.
        """
        # A catalog commit needs its state home to exist before the first
        # event is published, so observers never see a transient empty ledger.
        self.engine.ensure_state_ledger()
        return self.engine.submit_catalog(
            catalog, binding, name=name, after=tuple(after), owner=owner)

    def bind_memory(self, memory, *, replica: int = 0, stage: int = 0,
                    pools=("hbm",)):
        """Idempotently bind this Execution's real T08 ledger.

        Construction and an existing Execution from T14 use the same entry;
        a conflicting already-bound ledger is never replaced and returns None.
        Per-task scopes still allow a PD endpoint to carry a different ledger
        on the same Engine.
        """
        if memory is None:
            return self.memory
        if self.memory is not None and self.memory is not memory:
            return None
        if not self.engine.bind_memory(memory, pools=pools):
            return None
        self.memory = memory
        return memory

    def fragment_conductor(
        self, *, replica: int = 0, stage: int = 0,
        resource_bindings=None, pool_bindings=None, unit_conversions=None,
        state=None, detail: bool = False, cache_limit: int = 64,
        repeat_limit: int = 1_000_000, prefix: str | None = None,
        memory=None, memory_pools=("hbm",), memory_bindings=None,
        memory_owner=None, memory_storage=None, memory_ranges=None,
        memory_generation=None, memory_event=None, memory_host_bytes: float = 0.0,
        memory_unit: str | None = None,
    ):
        """One shared-context conductor for typed model/framework fragments.

        Declared fragment resource names are mapped through this stage's
        physical names (``cube``/``hbm``/...). Unknown names stay unresolved and
        raise ``fragment_resource_unbound`` at submit; they are never guessed.
        ``memory`` defaults to this Execution's ledger; pass the real PD endpoint
        ledger explicitly when one Engine serves two pools.
        """
        from .scheduling import FragmentConductor
        names = dict(self.stage_names.get((replica, stage), {}))
        names.update(dict(resource_bindings or {}))
        pools = dict(self.stage_pools.get((replica, stage), {}))
        pools.update(dict(pool_bindings or {}))
        if state is None:
            state = self.fragment_state
        ledger = memory if memory is not None else self.memory
        physical_memory_pools = tuple(
            pools.get(str(item), str(item)) for item in memory_pools)
        return FragmentConductor(
            self.engine, resource_bindings=names, pool_bindings=pools,
            unit_conversions=unit_conversions, state=state, detail=detail,
            memory=ledger, memory_replica=replica, memory_stage=stage,
            memory_pools=physical_memory_pools, memory_bindings=memory_bindings,
            memory_owner=memory_owner, memory_storage=memory_storage,
            memory_ranges=memory_ranges, memory_generation=memory_generation,
            memory_event=memory_event, memory_host_bytes=memory_host_bytes,
            memory_unit=memory_unit,
            cache_limit=cache_limit, repeat_limit=repeat_limit,
            prefix=prefix or f"fragment@{replica}:s{stage}:",
        )

    def submit_fragment(self, fragment, *, replica: int = 0, stage: int = 0,
                        **submit_kwargs):
        """Submit into the cached shared-context conductor for a stage.

        Use ``fragment_conductor`` directly when the assembly loop needs its
        own state/event/stream lifetime or non-default bindings.
        """
        key = (replica, stage)
        conductor = self.fragment_conductors.get(key)
        if conductor is None:
            conductor = self.fragment_conductor(replica=replica, stage=stage)
            self.fragment_conductors[key] = conductor
        return conductor.submit(fragment, **submit_kwargs)

    def stage_submit_boundary(self, replica: int = 0, stage: int = 0,
                              name: str = "host_submit", *, step=None):
        """Start event of a real host launch, or ``None``.

        Pass ``step=ScheduledIteration`` to answer "did *this* step submit a
        launch for this stage?". Without ``step`` the latest stage launch is
        used, which cannot distinguish a framework-only step from an earlier
        model step and must not be used as a no-launch proof.
        """
        if step is not None:
            matches = [event for rep, stg, event in step.host_submits
                       if rep == replica and stg == stage]
            return matches[-1] if matches else None
        launch = self.hosts.get((replica, stage))
        if launch is None:
            return None
        return self.engine.start_boundary(launch, name)

    def add_group(self, works, release_ms, *, external_inputs=None,
                  emitted_tokens=None, memory_bindings=None,
                  state_accesses=None):
        """Compile an EP/local-TP cohort before releasing any collective events."""
        graphs = bind_cohort_traffic(self.case, {r: work.ops for r, work in works.items()})
        from .local_projection_tp import lower_local_projection_tp_cohort
        graphs = lower_local_projection_tp_cohort(
            graphs, model=self.case.model, runtime=self.case.runtime,
            machine=self.case.machine)
        works = {r: replace(work, ops=graphs[r]) for r, work in works.items()}
        self._collectives = {}
        self._cohort_views = {
            int(rank): work.plan.requests
            for r, work in works.items()
            if work.plan.requests or work.mode != "graph"
            for stage in range(max(ONE, int(self.case.runtime.parallel.pp)))
            for rank in self.topology.ranks(r, stage)
        }

        def per_replica(value, replica):
            if isinstance(value, _AbcMapping) and replica in value:
                return value[replica]
            return value

        steps = {replica: self.add(
            work, replica, release_ms,
            external_inputs=(external_inputs or {}).get(replica),
            emitted_tokens=(emitted_tokens or {}).get(replica),
            memory_bindings=per_replica(memory_bindings, replica),
            state_accesses=per_replica(state_accesses, replica),
        ) for replica, work in works.items()}
        for key, participants in self._collectives.items():
            expected_count = len(works)
            first_meta = dict(participants[FIRST].context.op.metadata)
            if first_meta.get("group_axis") == "local_tp" and first_meta.get("group_ranks"):
                src_stage = int(first_meta.get("source_stage", participants[FIRST].context.op.stage))
                group_set = {int(rk) for rk in first_meta["group_ranks"]}
                expected_count = sum(
                    ONE for r in works
                    if set(self.topology.ranks(r, src_stage)) & group_set)
            if len(participants) != expected_count:
                raise ValueError(f"collective_participants_mismatch:{key}")
            if any(getattr(item.recipe, "algorithm", None) for item in participants):
                self._start_joint_collective(key, participants)
                continue
            native = any(getattr(item.recipe, "native", False)
                         for item in participants)
            if native:
                self._schedule_native_cohort(key, participants)
                continue
            if all(item.context.op.family == "mc2_transport"
                   for item in participants):
                programs = {item.replica: self.lower_program(
                    item.context, item.recipe) for item in participants}
                from .program import NativeProgram
                if all(isinstance(program, NativeProgram)
                       and hasattr(program.program, "start_with_handles")
                       for program in programs.values()):
                    if self._start_native_cohort(key, participants, programs):
                        continue
            source_stage = int(dict(participants[0].context.op.metadata).get(
                "source_stage", participants[0].context.op.stage))
            local_ready_by_rank = {}
            for item in participants:
                local_ready_by_rank.update({
                    rank: item.ready for rank in
                    self.topology.ranks(item.replica, source_stage)})
            observation = self._begin_joint_observation(
                participants[0].context.op, key, participants,
                local_ready_by_rank, source_stage)
            dependencies = tuple(item.ready for item in participants)
            ready = self.engine.submit(f"group:{key}:ready", after=dependencies, owner="expert_group")
            # Aggregate recipes keep their common cohort barrier.  A native
            # peer program with named send phases is launched by
            # _start_native_cohort above; a flat prework ready is never used
            # as a phase send.
            completions, promises = [], []
            for item in participants:
                completions.append(self._start_program(
                    item.context, item.recipe, item.name, ready,
                    owner="expert_group", replica=item.replica))
                promises.append(item.completion)

            def release(event, events=tuple(promises), observation=observation,
                        local_ready_by_rank=local_ready_by_rank,
                        completions=tuple(completions), ready=ready,
                        participants=tuple(participants),
                        source_stage=source_stage):
                if observation is not None:
                    # The common ready event is the actual cohort barrier; the
                    # local events remain the genuine per-rank upstream bounds.
                    observation["barrier_ms"] = ready.time
                    observation["participants"] = []
                    for item, completion in zip(participants, completions):
                        ranks = self.topology.ranks(item.replica, source_stage)
                        local = {rank: local_ready_by_rank[rank].time
                                 for rank in ranks}
                        done = {rank: completion.time for rank in ranks}
                        observation["participants"].append({
                            "replica": item.replica, "ranks": list(ranks),
                            "ready_ms": local, "completion_ms": done,
                            "wait_ms": {rank: ready.time - local[rank]
                                        for rank in ranks},
                            "parent_complete_ms": event.time,
                        })
                        observation["ready_ms_by_rank"].update(local)
                        observation["completion_ms_by_rank"].update(done)
                        observation["parent_complete_ms"][item.replica] = event.time
                    self._finish_joint_collective(observation)
                for item in events:
                    self.engine.resolve(item)

            self.engine.submit(f"group:{key}", after=completions,
                               on_complete=release, owner="expert_group")
        self._collectives = None
        self._cohort_views = None
        return steps

    def _begin_joint_observation(self, op, key, participants, ready_by_rank,
                                 source_stage):
        """Allocate one bounded record for an actual cohort rendezvous."""
        if self._joint_collective_trace is None:
            return None
        collective, expected_bytes, expected_dtype = _collective_native_basis(
            op, len(ready_by_rank))
        axis = dict(op.metadata).get("group_axis", "tp")
        occurrence = self._joint_collective_ordinals.get((op.family, axis), 0)
        self._joint_collective_ordinals[(op.family, axis)] = occurrence + 1
        link_by_rank, native_by_rank = {}, {}
        for item in participants:
            ranks = self.topology.ranks(item.replica, source_stage)
            _, native, _ = _collective_native_basis(
                item.context.op, len(ready_by_rank))
            value = (int(native) if native is not None
                     and native.denominator == 1 else None)
            link_value = item.context.native_work.get(
                "link", item.context.op.link_bytes)
            for rank in ranks:
                link_by_rank[rank] = link_value
                native_by_rank[rank] = value
        native_total = (sum(native_by_rank.values())
                        if native_by_rank
                        and all(value is not None
                                for value in native_by_rank.values())
                        else None)
        return {
            "op_id": op.op_id, "group": key,
            "family": op.family, "group_axis": axis,
            "collective": collective, "dtype": expected_dtype,
            "native_input_bytes": native_total,
            "native_input_bytes_by_rank": native_by_rank,
            "link_bytes": sum(link_by_rank.values()),
            "link_bytes_by_rank": link_by_rank,
            "source_stage": source_stage,
            "rank_count": len(ready_by_rank), "occurrence": occurrence,
            "participants": [], "ready_ms_by_rank": {},
            "completion_ms_by_rank": {}, "parent_complete_ms": {},
        }

    def _start_joint_collective(self, key, participants):
        """Bind one physical cohort while retaining each replica's readiness."""
        first = participants[0]
        algorithm = getattr(first.recipe, "algorithm", None)
        plan_options = collective_options(first.recipe, first.context.op)
        path_model = getattr(first.recipe, "path_model", None)
        op = first.context.op
        def declaration(item):
            return (
                dict(item.metadata).get("collective", item.family),
                dict(item.metadata).get("group_axis", "tp"),
                dict(item.metadata).get("source_stage", item.stage),
                dict(item.metadata).get("steps"),
                dict(item.metadata).get("peer_traffic"),
                dict(item.metadata).get("peer_volume_basis", "per_rank"),
            )
        for item in participants:
            if (getattr(item.recipe, "algorithm", None) != algorithm
                    or collective_options(item.recipe, item.context.op) != plan_options
                    or getattr(item.recipe, "path_model", None) != path_model
                    or declaration(item.context.op) != declaration(op)):
                raise ModelingError(f"joint_collective_declarations_differ:{key}")
            if any(value for role, value in item.context.native_work.items() if role != "link"):
                raise ModelingError(f"{algorithm}_collective_requires_transport_only_work")
        plan = self.topology.joint_plan(
            op, {item.replica: item.context.native_work.get("link", 0)
                 for item in participants}, algorithm=algorithm, **plan_options)
        plan.validate_path_model(path_model)
        source_stage = int(dict(op.metadata).get("source_stage", op.stage))
        ready_by_rank = {}
        for item in participants:
            timing = item.context.invocation_timing
            ready = item.ready
            if timing is not None and timing.startup_ms:
                ready = self.engine.submit(
                    item.name + ":device_startup", after=(ready,), latency=timing.startup_ms,
                    owner=f"replica:{item.replica}")
            ready_by_rank.update({rank: ready for rank in
                                  self.topology.ranks(item.replica, source_stage)})
        if self.readiness_report:
            from .collective_readiness import bind_pipeline_ready
            try:
                collective, expected_bytes, expected_dtype = _collective_native_basis(
                    op, len(ready_by_rank))
                context = bind_pipeline_ready(
                    self.readiness_report, ready_by_rank,
                    expected_collective=collective,
                    expected_bytes=(int(expected_bytes)
                                    if expected_bytes is not None
                                    and expected_bytes.denominator == 1
                                    else None),
                    expected_dtype=expected_dtype,
                    expected_group_size=len(ready_by_rank),
                    connection_id=dict(op.metadata).get("collective_connection_id"),
                    ordinal=dict(op.metadata).get("collective_call_ordinal"),
                    call_id=dict(op.metadata).get("collective_call_id"))
            except ModelingError as exc:
                self.readiness_unresolved.append({
                    "op_id": op.op_id, "group": key, "reason": str(exc)})
            else:
                self.readiness_observations.append({
                    "op_id": op.op_id, "group": key,
                    "ready_by_rank": tuple(sorted(ready_by_rank)),
                    "context": context,
                })
        completions = plan.start_participants(
            self.engine, first.name + ":cohort", ready_by_rank, owner="expert_group")
        observation = self._begin_joint_observation(
            op, key, participants, ready_by_rank, source_stage)
        pending = [len(participants)] if observation is not None else None
        for item in participants:
            ranks = self.topology.ranks(item.replica, source_stage)
            dependencies = [completions[rank] for rank in ranks]
            timing = item.context.invocation_timing
            minimum = max(item.context.minimum_ms, timing.minimum_ms if timing else 0)
            if minimum:
                dependencies.append(self.engine.submit(
                    item.name + ":observation", after=(item.ready,),
                    latency=minimum, owner=f"replica:{item.replica}"))

            def cohort_complete(event, item=item, ranks=ranks,
                                observation=observation, pending=pending):
                # The callback already runs on the physical cohort-complete
                # event.  Copy numeric boundaries; never retain Events.
                if observation is not None:
                    ready = {rank: ready_by_rank[rank].time for rank in ranks}
                    completion = {rank: completions[rank].time for rank in ranks}
                    observation["participants"].append({
                        "replica": item.replica, "ranks": list(ranks),
                        "ready_ms": ready, "completion_ms": completion,
                        "parent_complete_ms": event.time,
                    })
                    observation["ready_ms_by_rank"].update(ready)
                    observation["completion_ms_by_rank"].update(completion)
                    observation["parent_complete_ms"][item.replica] = event.time
                    pending[0] -= 1
                    if pending[0] == 0:
                        self._finish_joint_collective(observation)
                self.engine.resolve(item.completion)

            self.engine.submit(
                item.name + ":cohort_complete", after=dependencies,
                owner=f"replica:{item.replica}", on_complete=cohort_complete)
        return completions

    def _finish_joint_collective(self, observation: dict) -> None:
        """Reduce one completed cohort to bounded numeric observations."""
        observation["participants"].sort(key=lambda item: item["replica"])
        ready = [value for value in observation["ready_ms_by_rank"].values()
                 if value is not None]
        completion = [value for value in observation["completion_ms_by_rank"].values()
                      if value is not None]
        if ready:
            observation["rendezvous_ready_ms"] = observation.get(
                "barrier_ms", max(ready))
            observation["ready_spread_ms"] = max(ready) - min(ready)
            observation["critical_rank"] = max(
                observation["ready_ms_by_rank"],
                key=lambda rank: (observation["ready_ms_by_rank"][rank]
                                  if observation["ready_ms_by_rank"][rank] is not None
                                  else float("-inf")))
        if completion:
            observation["last_completion_ms"] = max(completion)
        from .participant_profile import select_call
        observation["source"] = select_call(
            self.participant_profile, primitive=observation["collective"],
            occurrence=observation["occurrence"],
            group_size=observation["rank_count"], dtype=observation["dtype"])
        stats = self.joint_collective_stats
        stats["observations"] += 1
        spread = observation.get("ready_spread_ms") or 0.0
        if spread > stats["max_ready_spread_ms"]:
            stats["max_ready_spread_ms"] = spread
            stats["max_rendezvous_delay_ms"] = spread
            stats["critical_rank"] = observation.get("critical_rank")
        trace = self._joint_collective_trace
        trace_key = (observation["op_id"], observation["group"])
        if trace_key in trace:
            trace[trace_key]["occurrences"] += 1
            trace[trace_key]["last_ready_spread_ms"] = spread
            trace[trace_key]["last_parent_complete_ms"] = max(
                observation["parent_complete_ms"].values(), default=None)
        else:
            observation["occurrences"] = 1
            if len(trace) >= self._joint_collective_limit:
                trace.pop(next(iter(trace)))
            trace[trace_key] = observation
        stats["retained"] = len(trace)

    def _outside_trace(self, engine) -> bool:
        prefix = getattr(engine, "_trace_prefix", None)
        if prefix is None:
            return False
        return any(
            (task.remaining or task.allocation_demands)
            and not task.event.name.startswith(prefix)
            for task in engine._tasks.values()
        )

    def _join_isolated_replay(self, engine, script_key, cached, token, complete) -> bool:
        steps, peaks = cached
        pending = engine._script_pending
        if pending is None:
            if not engine.replay_ready():
                return False
            engine.queue_script(script_key, steps, [(token, complete)], peaks)
            return True
        if engine._script_pending_key != script_key:
            return False
        pending[1].append((token, complete))
        return True

    def _seal_trace(self, engine, complete) -> None:
        key = self._trace_key
        shadows = self._trace_shadows
        self._trace_key = None
        self._trace_shadows = []
        if getattr(engine, "_trace_prefix", None) is None:
            engine.resolve(complete)
            return
        clean, steps, peaks = engine.end_trace()
        if key is not None and clean:
            Execution._isolated_scripts[key] = (steps, peaks)
            if len(Execution._isolated_scripts) > 1024:
                Execution._isolated_scripts.pop(next(iter(Execution._isolated_scripts)))
            # The recording rank already advanced time and its own counters.
            # Sibling ranks only receive the same increments.
            for token, shadow in shadows:
                self._apply_private_account(engine, steps, peaks, token)
                engine.resolve(shadow)
        else:
            for _token, shadow in shadows:
                engine.resolve(shadow)
        engine.resolve(complete)

    def _apply_private_account(self, engine, steps, peaks, token) -> None:
        for dt, busy, adds in steps:
            for name in busy:
                concrete = name.replace("@r*", token, 1)
                if concrete in engine.busy_ms:
                    engine.busy_ms[concrete] += dt
            for name, used in adds:
                concrete = name.replace("@r*", token, 1)
                if concrete in engine.serviced:
                    engine.serviced[concrete] += used
        for pool, unit, value in peaks:
            concrete = pool.replace("@r*", token, 1)
            if concrete not in engine.peak_allocated:
                continue
            pair = concrete, unit
            if value > engine.peak_unit_allocated.get(pair, 0.0):
                engine.peak_unit_allocated[pair] = value
            if value > engine.peak_allocated[concrete]:
                engine.peak_allocated[concrete] = value

    def _schedule_native_cohort(self, key, participants):
        """Defer the whole native cohort until its first genuine ready.

        The first callback prepares the bounded placeholder registry for this
        layer only, then each participant materializes its own program at its
        own ready Event.  No future layer contributes peer tasks before its
        first source is ready.
        """
        started = [False]
        def begin(_event):
            if started[0]:
                return
            started[0] = True
            programs = {item.replica: self.lower_program(
                item.context, item.recipe) for item in participants}
            from .program import NativeProgram
            if not all(isinstance(program, NativeProgram)
                       and hasattr(program.program, "start_with_handles")
                       for program in programs.values()):
                raise ModelingError(
                    f"native_peer_plan_requires_executable_cohort:{key}")
            if not self._start_native_cohort(key, participants, programs,
                                             defer=True):
                raise ModelingError(
                    f"native_peer_plan_requires_executable_cohort:{key}")
        for item in participants:
            self.engine.submit(
                f"group:{key}:native_wait:r{item.replica}",
                after=(item.ready,), owner="expert_group",
                on_complete=begin)
        return True

    def _start_native_cohort(self, key, participants, programs, *, defer=False):
        """Two-phase native launch for one ordinary EP cohort."""
        source_stage = int(dict(participants[0].context.op.metadata).get(
            "source_stage", participants[0].context.op.stage))
        launches = []
        for item in participants:
            program = programs[item.replica]
            rank = self.participant_rank_index(item.replica, source_stage)
            if rank is None:
                return False
            launch = {
                "program": program,
                "name": f"group:{key}:r{item.replica}",
                "ready": item.ready,
                "rank_index": rank,
                "peer_count": int(getattr(program.program, "peer_count", 0)),
                "node_ranks": int(getattr(self.case.machine, "ranks_per_node", 0)),
                "resource_names": self.native_resource_names(
                    item.context, program, rank=rank,
                    replica=item.replica),
                "transport": self.native_transport(
                    item.context.op, item.replica),
            }
            launches.append(launch)
        result = launch_native_cohort(
            self.engine, launches, name_prefix=f"group:{key}",
            owner="expert_group", defer=defer)
        for item in participants:
            name = f"group:{key}:r{item.replica}"
            parent = result["parents"][name]
            self.engine.submit(
                name + ":release", after=(parent,),
                owner=f"replica:{item.replica}",
                on_complete=lambda _event, promise=item.completion:
                self.engine.resolve(promise))
        self.native_group_joins[key] = result["join"]
        raw = dict(participants[0].context.op.metadata).get("mc2_phase", {})
        payload_edges = sum(
            int(phases.get("payload_exchange", {}).get("edges", 0))
            for phases in result["transport_by_rank"].values())
        count_edges = sum(
            int(phases.get("count_status_exchange", {}).get("edges", 0))
            for phases in result["transport_by_rank"].values())
        payload_frames = sum(
            int(phases.get("payload_exchange", {}).get("frames", 0))
            for phases in result["transport_by_rank"].values())
        merged_work = {}
        for name, work in result["work_by_rank"].items():
            merged = {pipe: amount for pipe, amount in work
                      if "link" not in str(pipe).lower()}
            for pipe, amount in result["transport_work_by_rank"].get(
                    name, {}).items():
                merged[pipe] = merged.get(pipe, 0.0) + float(amount)
            merged_work[name] = tuple(sorted(merged.items()))
        self.native_group_launches[key] = {
            "op_id": key,
            "kind": str(raw.get("kind", "")),
            "launch_mode": "cohort",
            "ranks": tuple(sorted(result["rank_indices"].values())),
            "payload_basis": dict(result["payload_basis"]),
            "payload_packet_bytes": dict(result["payload_packet_bytes"]),
            "payload_frames": dict(result["payload_frames"]),
            "link_by_phase_by_rank": dict(result["link_by_phase_by_rank"]),
            "work_by_rank": merged_work,
            "transport_by_rank": {
                name: dict(phases)
                for name, phases in result["transport_by_rank"].items()},
            "transport_work_by_rank": {
                name: dict(work)
                for name, work in result["transport_work_by_rank"].items()},
            "transport_tx_bytes": float(result["transport_tx_bytes"]),
            "transport_rx_bytes": float(result["transport_rx_bytes"]),
            "transport_frames": int(result["transport_frames"]),
            "transport_edges": payload_edges,
            "total_edges": int(result["transport_edges"]),
            "payload_edges": payload_edges,
            "count_edges": count_edges,
            "directed_edges": (payload_edges or int(result["directed_edges"])),
            "directed_frames": (payload_frames
                                or int(result["directed_frames"])),
            "protocol_edges": int(result["protocol_edges"]),
            "unresolved_edges": int(result["unresolved_edges"]),
            "partial": tuple(result["partial"]),
        }
        return True

    def participant_peer_ready(self, replica, source_stage, ready_by_rank):
        """Actual peer ready Events for one participant, keyed by EP rank.

        These are the same shared-Engine handles exposed by
        ``CollectiveParticipant.ready``; no skew, timestamp or inferred wait is
        introduced.  A native peer program uses each handle only for its named
        peer receive/check/retire edge.
        """
        local_ranks = set(self.topology.ranks(replica, source_stage))
        for group in self.topology.groups("ep", replica, source_stage):
            if local_ranks & set(group):
                return {index: ready_by_rank[rank]
                        for index, rank in enumerate(group)
                        if rank not in local_ranks}
        return {rank: event for rank, event in ready_by_rank.items()
                if rank not in local_ranks}

    def participant_rank_index(self, replica, source_stage):
        """Local EP rank index for one participant (TP1/MC2 ingress)."""
        local_ranks = tuple(self.topology.ranks(replica, source_stage))
        for group in self.topology.groups("ep", replica, source_stage):
            for rank in local_ranks:
                if rank in group:
                    return group.index(rank)
        return None

    def _peer_ready_for(self, item, local_ready_by_rank, source_stage):
        return self.participant_peer_ready(
            item.replica, source_stage, local_ready_by_rank)

    def _topology_class_resource(self, rank, pipe):
        """Actual TX port for this rank/class, or None when not port-scoped."""
        rank = int(rank)
        kind = "roce" if pipe == "link_roce" else "hccs"
        links = list(getattr(self.topology, "links", {}).values())
        preferred = [link for link in links
                     if (kind == "roce") == ("roce" in link.name.lower()
                                             or "inter" in link.name.lower())]
        for link in preferred or links:
            if rank not in link.members:
                continue
            if link.semantics == "per_pair":
                peer = next((member for member in link.members
                             if member != rank), None)
                if peer is None:
                    continue
                return self.topology.resource(link, rank, peer)
            if link.semantics in {"per_port", "fully_connected", "mesh"}:
                return f"{self.topology.namespace}link:{link.name}:tx:port{rank}"
        return None

    def native_transport(self, op, replica):
        """Whole actual EP membership and source cells for this launch.

        TP8/DP1 has eight physical EP members even though the graph has one DP
        replica.  The replica owns the transport cells whose source rank is
        local; peer ranks are endpoints, not cloned expert/FFN compute.
        """
        metadata = dict(op.metadata)
        source_stage = int(metadata.get("source_stage", op.stage))
        local_ranks = tuple(self.topology.ranks(replica, source_stage))
        groups = tuple(self.topology.groups("ep", replica, source_stage))
        group = next(
            (members for members in groups
             if set(members) & set(local_ranks)), ())
        if not group:
            group = local_ranks
        sources = tuple(index for index, rank in enumerate(group)
                        if rank in local_ranks)
        destination_memory = {
            int(rank): self.destination_window_resource(int(rank), source_stage)
            for rank in group}
        return {"group": tuple(group), "sources": sources,
                "route": self.topology.route_message,
                "destination_memory": destination_memory}

    def native_resource_names(self, context, program, *, rank=None,
                              replica=None):
        """Public launch-resource binding for canonical cohort drivers."""
        return self._native_resource_names(
            context, program, rank=rank, replica=replica)

    def _native_resource_names(self, context, program, *, rank=None,
                               replica=None):
        """Bind native pipe names to this rank's physical stage resources."""
        inner = getattr(program, "program", program)
        native_names = getattr(inner, "names", {}) or {}
        available = dict(getattr(context, "resource_names", {}) or {})
        bound = {str(role): str(physical)
                 for role, physical in available.items()}
        for pipe, native_name in native_names.items():
            physical = available.get(pipe) or available.get(native_name)
            if rank is not None and pipe in {"link", "link_hccs", "link_roce"}:
                route = self._topology_class_resource(rank, str(pipe))
                if route is not None:
                    physical = route
            if physical is not None:
                bound[str(pipe)] = str(physical)
        if rank is not None:
            # A source phase names the fabric class directly even when the
            # stage resource table exposes only per-rank topology ports.
            for pipe in ("link", "link_hccs", "link_roce"):
                if pipe in bound:
                    continue
                route = self._topology_class_resource(rank, pipe)
                if route is not None:
                    bound[pipe] = route
        return bound

    @staticmethod
    def _build_launch_view(inner, rank_index, *, node_ranks=None,
                           peer_classes=None, peer_weights=None,
                           intra_latency_ms=None, inter_latency_ms=None,
                           resource_names=None):
        """Immutable per-launch rank/topology view; cached program untouched."""
        from .mc2_program import Mc2LaunchView
        peer_count = int(getattr(inner, "peer_count", 0) or 0)
        if peer_count <= 0 or rank_index is None:
            return None
        rank = int(rank_index)
        if not 0 <= rank < peer_count:
            raise ModelingError(f"native_launch_rank_out_of_range:{rank}")
        source = getattr(inner, "view", None)
        labels = tuple(item for item in range(peer_count) if item != rank)
        classes = tuple(peer_classes or ())
        weights = tuple(peer_weights or ())
        if source is not None and source.rank == rank:
            if not classes:
                classes = tuple(source.peer_classes)
            if not weights:
                weights = tuple(source.peer_weights)
        node = int(node_ranks if node_ranks is not None else
                   (getattr(source, "node_ranks", 0) if source else 0))
        intra = (getattr(source, "intra_latency_ms", 0)
                 if intra_latency_ms is None and source
                 else (intra_latency_ms or 0))
        inter = (getattr(source, "inter_latency_ms", 0)
                 if inter_latency_ms is None and source
                 else (inter_latency_ms or 0))
        return Mc2LaunchView(
            rank=rank, peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=node,
            intra_latency_ms=intra, inter_latency_ms=inter,
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(resource_names or {}).items())))

    @staticmethod
    def _phase_scope(program, mapping):
        """Normalise a flat peer map to MOE10's named-phase mapping."""
        if mapping is None:
            return None
        inner = getattr(program, "program", program)
        phases = getattr(inner, "phases", None)
        link_by_phase = getattr(inner, "link_by_phase", None)
        if not phases or not link_by_phase:
            return dict(mapping)
        names = {phase.name for phase in phases}
        if mapping and all(isinstance(key, str) and key in names
                           for key in mapping):
            return dict(mapping)
        peers = dict(mapping)
        return {phase.name: dict(peers) for phase in phases
                if phase.name in link_by_phase}

    def start_native_participant_with_handles(self, program, name, ready, *,
                                              owner="", local_ready=None,
                                              peer_ready=None, peer_send=None,
                                              rank_index=None, launch=None,
                                              source_send=None,
                                              send_placeholders=None,
                                              resource_names=None,
                                              transport=None):
        """Start a native peer program and return its actual phase handles.

        ``peer_ready`` and ``peer_send`` accept either the flat
        ``{ep_rank: Event}`` form (applied to every source link phase) or
        MOE10's exact ``{phase_name: {ep_rank: Event}}`` form.  The returned
        ``handles`` are the program's launch-local peer send/receive/check/
        retire and phase-complete Events on this same Engine.
        """
        from .program import NativeProgram
        phase_ready = self._phase_scope(program, peer_ready)
        phase_send = self._phase_scope(program, peer_send)
        start = local_ready if local_ready is not None else ready
        if isinstance(program, NativeProgram):
            inner = program.program
            if launch is None and rank_index is not None:
                launch = Execution._build_launch_view(
                    inner, rank_index, resource_names=resource_names)
            if hasattr(inner, "start_with_handles"):
                return program.start_with_handles(
                    self.engine, name, start, peer_ready=phase_ready,
                    peer_send=phase_send, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    transport=transport)
            if peer_send is not None:
                raise ModelingError(
                    "native_peer_send_requires_start_with_handles")
            return program.start(self.engine, name, start,
                                 peer_ready=phase_ready), {}
        event = self._start_engine_program(
            program, self.engine, name, ready, owner,
            local_ready=local_ready, peer_ready=peer_ready,
            peer_send=peer_send)
        return event, {}

    def start_native_participant(self, program, name, ready, *, owner="",
                                 local_ready=None, peer_ready=None,
                                 peer_send=None, rank_index=None,
                                 launch=None, source_send=None,
                                 send_placeholders=None, transport=None):
        """Start a lowered native program on this Execution's shared Engine.

        MOE10 calls this with the same Engine Events used by
        ``CollectiveParticipant``.  A ``NativeProgram`` starts at its own real
        ``local_ready`` and receives phase-scoped ``peer_ready``/``peer_send``
        for its named peer receive/check/retire edges.  Any non-native program
        keeps the supplied common ``ready`` barrier.
        """
        event, _handles = self.start_native_participant_with_handles(
            program, name, ready, owner=owner, local_ready=local_ready,
            peer_ready=peer_ready, peer_send=peer_send,
            rank_index=rank_index, launch=launch, source_send=source_send,
            send_placeholders=send_placeholders, transport=transport)
        return event

    @staticmethod
    def _start_engine_program(program, engine, name, ready, owner,
                              local_ready=None, peer_ready=None,
                              peer_send=None, rank_index=None, launch=None,
                              source_send=None, send_placeholders=None,
                              resource_names=None, transport=None):
        """Start one lowered program on the caller's shared Engine.

        A native peer program starts at its own real ready and receives the
        actual peer Events.  Every other program keeps the common cohort
        barrier, so ordinary aggregate recipes are unchanged.
        """
        from .program import NativeProgram
        if isinstance(program, NativeProgram) and (
                peer_ready is not None or peer_send is not None):
            start = local_ready if local_ready is not None else ready
            inner = program.program
            if launch is None and rank_index is not None:
                launch = Execution._build_launch_view(
                    inner, rank_index, resource_names=resource_names)
            phase_ready = Execution._phase_scope(program, peer_ready)
            phase_send = Execution._phase_scope(program, peer_send)
            if peer_send is not None:
                if not hasattr(program.program, "start_with_handles"):
                    raise ModelingError(
                        "native_peer_send_requires_start_with_handles")
                event, _handles = program.start_with_handles(
                    engine, name, start, peer_ready=phase_ready,
                    peer_send=phase_send, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    transport=transport)
                return event
            return program.start(engine, name, start, peer_ready=phase_ready,
                                 launch=launch, source_send=source_send,
                                 transport=transport)
        start = ready
        if local_ready is not None and local_ready is not ready:
            start = engine.submit(name + ":cohort_gate", after=(ready,))
        if isinstance(program, NativeProgram):
            return program.start(engine, name, start, transport=transport)
        return program.start(engine, name, start, owner=owner)

    def _record_single_native(self, context, program, replica, name,
                              transport=None):
        """Retain an actual non-cohort native launch as public evidence.

        A representative single-replica graph still executes its native peer
        program on this Execution.  It is a real launch, but it has no
        placeholder cohort, so it is reported separately from a complete
        cross-DP rendezvous instead of being labelled by family.
        """
        from .program import NativeProgram
        if not isinstance(program, NativeProgram):
            return
        inner = program.program
        if not hasattr(inner, "start_with_handles"):
            return
        key = context.op.op_id
        if key in self.native_group_launches:
            return
        meta = dict(context.op.metadata)
        raw = meta.get("mc2_phase", {})
        source_stage = int(meta.get("source_stage", context.op.stage))
        rank = (self.participant_rank_index(replica, source_stage)
                if replica is not None else None)
        if rank is None:
            rank = 0
        frames_for = getattr(inner, "frames_for", None)
        frames = (tuple(frames_for(rank)) if callable(frames_for) else
                  tuple(getattr(getattr(inner, "view", None),
                                "payload_frames", ()) or ()))
        basis_for = getattr(inner, "payload_basis_for", None)
        basis = (basis_for(rank) if callable(basis_for) else
                 str(getattr(getattr(inner, "view", None),
                             "payload_basis", "unknown")))
        packet = int(getattr(getattr(inner, "view", None),
                            "payload_packet_bytes", 0) or 0)
        links_for = getattr(inner, "link_by_phase_for", None)
        links = (links_for(rank) if callable(links_for)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        work_for = getattr(inner, "work_for_rank", None)
        work = dict(work_for(rank) if callable(work_for)
                    else tuple(getattr(program, "work", ()) or ()))
        transport_details = {}
        transport_work = {}
        if transport is not None and hasattr(inner, "transport_account"):
            payload_account, count_account = inner.transport_account(transport)
            has_count = any(getattr(item, "name", "") == "count_status_exchange"
                            for item in getattr(inner, "phases", ()))
            phases = (("payload_exchange", payload_account),
                      ("count_status_exchange", count_account))
            for phase_name, account in phases:
                if phase_name == "count_status_exchange" and not has_count:
                    continue
                if not account.get("edges"):
                    continue
                transport_details[phase_name] = {
                    "frames": int(account.get("frames", 0)),
                    "edges": int(account.get("edges", 0)),
                    "tx_bytes": float(account.get("tx_bytes", 0.0)),
                    "rx_bytes": float(account.get("rx_bytes", 0.0)),
                    "basis": str(getattr(inner, "transport_basis", basis)),
                    "group": list(account.get("group") or ()),
                    "sources": list(account.get("sources") or ()),
                    "count": phase_name == "count_status_exchange",
                }
                for pipe, amount in account.get("work", {}).items():
                    transport_work[pipe] = (
                        transport_work.get(pipe, 0.0) + float(amount))
        # Replace the logical per-rank link estimate with the actual coupled
        # TX/RX hop demand when a real route was scheduled.
        if transport_work:
            work = {pipe: amount for pipe, amount in work.items()
                    if "link" not in str(pipe).lower()}
            for pipe, amount in transport_work.items():
                work[pipe] = work.get(pipe, 0.0) + amount
        total_edges = sum(item["edges"] for item in transport_details.values())
        transport_frames = sum(item["frames"] for item in transport_details.values())
        payload_edges = int(transport_details.get(
            "payload_exchange", {}).get("edges", 0))
        count_edges = int(transport_details.get(
            "count_status_exchange", {}).get("edges", 0))
        self.native_group_launches[key] = {
            "op_id": key, "kind": str(raw.get("kind", "")),
            "launch_mode": "single_program",
            "ranks": (rank,),
            "payload_basis": {name: str(basis)},
            "payload_packet_bytes": {name: packet},
            "payload_frames": {name: dict(frames)},
            "link_by_phase_by_rank": {
                name: {phase: {str(pipe): float(amount)
                               for pipe, amount in values.items()}
                       for phase, values in links.items()}},
            "work_by_rank": {name: tuple(sorted(work.items()))},
            "transport_by_rank": {name: transport_details},
            "transport_work_by_rank": {name: transport_work},
            "transport_tx_bytes": sum(
                item["tx_bytes"] for item in transport_details.values()),
            "transport_rx_bytes": sum(
                item["rx_bytes"] for item in transport_details.values()),
            "transport_frames": transport_frames,
            "transport_edges": payload_edges,
            "total_edges": total_edges,
            "payload_edges": payload_edges,
            "count_edges": count_edges,
            "directed_edges": (transport_details.get(
                "payload_exchange", {}).get("edges") or len(frames)),
            "directed_frames": (transport_details.get(
                "payload_exchange", {}).get("frames")
                or sum(int(count) for _peer, count in frames)),
            "protocol_edges": 0,
            "unresolved_edges": 0,
            "partial": (),
        }

    def _program_inputs(self, context, recipe, engine, replica, op, owner):
        """Virtual-time-free lowering inputs of one operator occurrence."""
        transport = (
            self.native_transport(op, replica)
            if replica is not None and getattr(recipe, "native", False)
            else None)
        live_rates = tuple(
            (role, context.rates.get(res_name))
            for role, res_name in sorted(context.resource_names.items())
            if res_name in context.rates
        )
        inv_t = context.invocation_timing
        do_compact = not self.legacy_diagnostics and (
            not engine.trace.maxlen or self._compact_step
        )
        prog_key = (
            owner, do_compact, _freeze_key(recipe), op.family, op.recipe, op.phase, op.stage,
            op.dtype, op.layout, op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            self._op_key(op), context.setup_ms, context.minimum_ms,
            (inv_t.startup_ms, inv_t.minimum_ms) if inv_t is not None else None,
            _freeze_key(context.joint_native_service),
            tuple(sorted(context.work.items())),
            live_rates,
            context.group_buffers,
            getattr(context.physical_mechanisms, "signature", None),
        )
        token = _rank_token(context)
        replayable = (
            Execution.schedule_replay and token is not None
            and not op.link_bytes and not getattr(recipe, "algorithm", None)
            and dict(op.metadata).get("group_axis") not in {"ep", "local_tp"}
        )
        return transport, prog_key, do_compact, replayable

    def _lower_operator_program(self, context, recipe, engine, replica, name,
                                prog_key, do_compact):
        """Return ``((program, mapped, services), from_cache)`` for one operator.

        The lowered program is a pure function of the operator declaration and
        its frozen execution options, so it is cached per Engine and, when an
        explicit evaluation context exists, per evaluation.  Mapped records are
        replayed on every use; the program itself is never mutated.
        """
        hit = self._program_cache.get(prog_key)
        if hit is None and self._shared_eval_cache is not None:
            hit = self._shared_eval_cache.get(prog_key)
            if hit is not None:
                self._program_cache[prog_key] = hit
        if hit is not None:
            return hit, True
        mapped_list = []
        svc_list = []
        orig_rec = context.record_mapping
        orig_svc = context.record_service

        def cap_map(rec):
            mapped_list.append(deepcopy(rec))
            if orig_rec is not None:
                orig_rec(rec)

        def cap_svc(*args, **kwargs):
            svc_list.append((deepcopy(args), deepcopy(kwargs)))
            if orig_svc is not None:
                orig_svc(*args, **kwargs)

        run_ctx = replace(
            context,
            record_mapping=cap_map if orig_rec is not None else None,
            record_service=cap_svc if orig_svc is not None else None,
        )
        program = self.lower_program(run_ctx, recipe)
        program = self._apply_resource_service_program(context, program)
        if do_compact:
            from .program import coalesce_execution_program
            program = coalesce_execution_program(program)
        entry = (program, tuple(mapped_list), tuple(svc_list))
        if len(self._program_cache) >= PROGRAM_CACHE_ENTRY_LIMIT:
            self._program_cache.pop(next(iter(self._program_cache)))
        self._program_cache[prog_key] = entry
        if self._shared_eval_cache is not None:
            self._shared_eval_cache.put(prog_key, entry)
        return entry, False

    def _program_uses_fifo(self, context, engine) -> bool:
        """True when FIFO arbitration could order this program by sequence."""
        names = engine.fifo_resource_names()
        if not names:
            return False
        if engine._fifo_live > 0:
            return True
        for resource in context.resource_names.values():
            if resource in names:
                return True
        return False

    def _start_program(self, context, recipe, name, ready, *, owner,
                       local_ready=None, peer_ready=None, rank_index=None,
                       launch=None, source_send=None,
                       send_placeholders=None, replica=None,
                       memory_ledger=None, memory_pools=None,
                       memory_bindings=None, on_resolve=None, begin_after=None):
        """Keep per-core programs resident only while their operator can run."""
        engine = self.engine
        entry = {"ready": ready}
        inherited = local_ready if local_ready is not None else ready
        stage = int(context.op.stage)

        # Eager template preparation and cache lookup.  A pure static program is
        # admitted here without a ``:begin`` row or a Python completion callback,
        # but only while no live row can be ordered by a FIFO resource: eager
        # submission otherwise changes task sequence numbers and therefore FIFO
        # admission order (a demonstrated behavioural divergence).
        if (begin_after is None and on_resolve is None and memory_ledger is None
                and not memory_bindings and not Execution.schedule_replay
                and len(engine._tasks) < INLINE_FRONTIER_TASK_CAP
                and not self._program_uses_fifo(context, engine)):
            op = context.op
            transport, prog_key, do_compact, replayable = self._program_inputs(
                context, recipe, engine, replica, op, owner)
            if not replayable:
                (program, mapped_recs, svc_calls), _from_cache = (
                    self._lower_operator_program(
                        context, recipe, engine, replica, name, prog_key,
                        do_compact))
                from .program import is_pure_static_program
                if is_pure_static_program(program):
                    if mapped_recs and context.record_mapping is not None:
                        for mrec in mapped_recs:
                            context.record_mapping(mrec)
                    if svc_calls and context.record_service is not None:
                        for sargs, skwargs in svc_calls:
                            context.record_service(*sargs, **skwargs)
                    native_resources = self._native_resource_names(context, program)
                    self._record_single_native(
                        context, program, replica, name, transport=transport)
                    with engine.memory_scope(
                            memory_ledger,
                            replica=(0 if replica is None else replica),
                            stage=stage,
                            pools=(memory_pools if memory_pools is not None
                                   else ("hbm",)),
                            bindings=memory_bindings):
                        return self._start_engine_program(
                            program, engine, name, inherited, owner,
                            local_ready=local_ready, peer_ready=peer_ready,
                            rank_index=rank_index, launch=launch,
                            source_send=source_send,
                            send_placeholders=send_placeholders,
                            resource_names=native_resources,
                            transport=transport)

        complete = engine.future(name + ":program_complete")

        def begin(_):
            op = context.op
            ready = entry["ready"]
            transport, prog_key, do_compact, replayable = self._program_inputs(
                context, recipe, engine, replica, op, owner)
            token = _rank_token(context)
            if replayable:
                script_key = _isolated_script_key(prog_key, context, token)
                # Later layers run one rank for real so latency fill is recomputed
                # at the current virtual time. Replaying stored steps is not
                # bit-exact once absolute time changes a latency remainder.
                if (engine._trace_prefix is not None and engine._trace_private
                        and engine._trace_clean and engine._trace_token != token
                        and not self._outside_trace(engine)):
                    self._trace_shadows.append((token, complete))
                    self.replay_counts["shadow"] += 1
                    return
                if engine._trace_prefix is None and engine.replay_ready():
                    engine.begin_trace(name, token)
                    self._trace_key = script_key
                    self._trace_shadows = []
                    self.replay_counts["record"] += 1
                else:
                    self.replay_counts["real"] += 1
            else:
                self.replay_counts["real"] += 1

            recording = getattr(engine, "_trace_prefix", None) == name

            def finished(_event=None):
                if recording and getattr(engine, "_trace_prefix", None) == name:
                    self._seal_trace(engine, complete)
                else:
                    engine.resolve(complete)
                if on_resolve is not None:
                    on_resolve()

            (program, mapped_recs, svc_calls), _from_cache = (
                self._lower_operator_program(
                    context, recipe, engine, replica, name, prog_key,
                    do_compact))
            if mapped_recs and context.record_mapping is not None:
                for mrec in mapped_recs:
                    context.record_mapping(mrec)
            if svc_calls and context.record_service is not None:
                for sargs, skwargs in svc_calls:
                    context.record_service(*sargs, **skwargs)
            native_resources = self._native_resource_names(context, program)
            self._record_single_native(
                context, program, replica, name, transport=transport)
            finish = self._start_engine_program(
                program, engine, name, ready, owner,
                local_ready=local_ready, peer_ready=peer_ready,
                rank_index=rank_index, launch=launch,
                source_send=source_send,
                send_placeholders=send_placeholders,
                resource_names=native_resources, transport=transport)
            engine.on_complete(finish, finished)

        # Host submission and graph dependencies are already declared. Expanding
        # every future layer's per-core pipelines here multiplies inactive state
        # by the device's core count without changing any physical work.

        def begin_with_binding(event):
            with engine.memory_scope(
                    memory_ledger, replica=(0 if replica is None else replica),
                    stage=stage,
                    pools=(memory_pools if memory_pools is not None
                           else ("hbm",)),
                    bindings=memory_bindings):
                begin(event)

        if begin_after is None:
            engine.submit(name + ":begin", after=(inherited,), owner=owner,
                          on_complete=begin_with_binding, native_hook=True)
        else:
            # One dependency-satisfied boundary owns both the graph readiness
            # and the expansion trigger, so a caller without an operator
            # observer never pays for a separate zero-work ready task.
            boundary = engine.submit(
                name + ":begin", after=tuple(begin_after), owner=owner,
                on_complete=begin_with_binding, native_hook=True)
            entry["ready"] = boundary
        return complete

    @staticmethod
    def lower_program(context, recipe):
        recipe = context.program_recipe or recipe
        if context.joint_native_service is not None:
            from .program import KernelProgram, Stage
            return KernelProgram((Stage("joint_native_service", context.work),))
        from .invocation_binding import lower_invocation
        return lower_invocation(recipe, context)

    def transport_options_for(self, op, replica: int = 0, scope: str = "none"):
        """Return the exact TransportOptions bound for this occurrence."""
        return self._transport_options.get((str(scope), op.op_id, replica))

    def _apply_resource_service_program(self, context, program):
        binding = dict(context.op.metadata).get("resource_service")
        if binding is None:
            return program
        if not isinstance(binding, _AbcMapping):
            raise ModelingError("resource_service_binding_requires_mapping")
        service_id = binding.get("service_id")
        service = self.resource_services.get(service_id)
        if service is None:
            raise ModelingError(f"resource_service_binding_unknown:{service_id}")
        if service.scope != "independent_resource" or service.kind != "resource_rate":
            raise ModelingError("resource_service_binding_requires_independent_rate")
        role = binding.get("resource_role")
        physical = context.resource_names.get(role)
        if physical is None:
            raise ModelingError(f"resource_service_role_unbound:{role}")
        work_unit = binding.get("work_unit", service.work_unit)
        expected = context.work_units.get(role, service.work_unit)
        if service.work_unit != work_unit or expected != work_unit:
            raise ModelingError(
                f"resource_service_work_unit_mismatch:{service.service_id}:"
                f"{service.work_unit}:{work_unit}:{expected}")
        if service.phase and binding.get("phase")                 and service.phase != binding.get("phase"):
            raise ModelingError(
                f"resource_service_phase_mismatch:{service.service_id}")
        from .program import KernelProgram
        if not isinstance(program, KernelProgram):
            raise ModelingError("resource_service_requires_kernel_program")
        # A Stage task is one physical owner. Apply the declared per-owner
        # rate, never copy an aggregate rate to every owner.
        cap = service.per_owner_rate / 1000.0
        stages = []
        for stage in program.stages:
            if physical in stage.work:
                caps = dict(stage.caps)
                caps[physical] = min(caps.get(physical, cap), cap)
                stage = replace(stage, caps=caps)
            stages.append(stage)
        body = KernelProgram(tuple(stages), program.minimum_ms)
        if service.fixed_us > 0:
            return _ServiceSetupProgram(body, service.fixed_us / 1000.0)
        return body

    def prepare(self, op, replica=ZERO):
        # Directly submitted blocking work must not reach the Engine either.
        from .unmodeled_gate import assert_timing_supported
        assert_timing_supported(None, (op,), entry="execution_prepare")
        geom = self.stage_geometry[replica, op.stage]
        startup_mode = self.case.runtime.collective_startup_mode
        prep_key = (
            id(self.recipes), _freeze_key(geom), startup_mode, op.stage,
            op.family, op.recipe, op.phase, op.dtype, op.layout,
            op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            self._op_key(op),
            getattr(geom, "physical_mechanisms").signature
            if getattr(geom, "physical_mechanisms", None) is not None else None,
        )
        if prep_key in self._prepare_cache:
            cached_op, meta_diff, recipe = self._prepare_cache[prep_key]
            meta = op.metadata
            if meta_diff is not None:
                patched = _meta_patch(dict(op.metadata), meta_diff)
                meta = tuple(sorted(patched.items()))
            return replace(
                cached_op, op_id=op.op_id, layer=op.layer, deps=op.deps,
                requests=op.requests, stream=op.stream, notes=op.notes,
                metadata=meta,
            ), recipe
        res_op, recipe = self.recipes.prepare(op, geom)
        from .collective_waves import startup_waves_recipe
        recipe = startup_waves_recipe(res_op, recipe, startup_mode)
        stage_dies = self.stage_dies.get((replica, res_op.stage))
        if stage_dies:
            from .matrix_precision import resolve_matrix_precision
            res_op = resolve_matrix_precision(
                res_op, devices=stage_dies,
                policy=self.case.runtime.matrix_compute_policy, recipe=recipe)
        orig_meta = dict(op.metadata)
        res_meta = dict(res_op.metadata)
        meta_diff = _meta_diff(orig_meta, res_meta) if orig_meta != res_meta else None
        if len(self._prepare_cache) >= 512:
            self._prepare_cache.pop(next(iter(self._prepare_cache)))
        self._prepare_cache[prep_key] = (res_op, meta_diff, recipe)
        return res_op, recipe

    def destination_window_resource(self, destination_rank: int, stage: int) -> str | None:
        """Return the declared resident memory resource for one represented rank.

        The topology defines which physical rank is the first die of each
        ordinary execution replica. Only those declared representative ranks
        own a resident ``memory@rN:sS`` resource; other physical EP endpoints are
        network-only placeholders and are not given invented local capacity.
        """
        for (replica, owner_stage), names in self.stage_names.items():
            if owner_stage != stage:
                continue
            ranks = self.topology.ranks(replica, stage)
            if ranks and ranks[0] == destination_rank:
                return names.get("hbm")
        return None

    def _stage_context(self, key, dies):
        replica, stage = key
        profile = self.stage_profiles.get(key)
        if profile is None:
            names = self.stage_names[key]
            return RecipeContext(
                None, {}, names, self.resource_rates,
                dies[FIRST].aic_count,
                pool_names={pool: f"{self.namespace}{pool}@r{replica}:s{stage}"
                            for pool in dict(dies[FIRST].buffers)},
                resource_units={role: dies[FIRST].units(role) for role in names},
                compute_peaks={dtype: min(d.peak(dtype) for d in dies)
                               for dtype, _ in dies[FIRST].compute_peaks
                               if dtype not in dict(dies[FIRST].compute_peak_aliases)},
                pool_units={pool: dies[FIRST].pool_units(pool)
                            for pool in dict(dies[FIRST].buffers)},
                pool_capacities={pool: min(dict(d.buffers)[pool] for d in dies)
                                 for pool in dict(dies[FIRST].buffers)},
                record_mapping=self.cost.record_mapping,
                service_modes={mode: (role, min(dict(d.service_modes)[mode][SECOND] for d in dies))
                               for mode, (role, _) in dies[FIRST].service_modes},
                record_service=self.cost.record_service,
                work_units={**NATIVE_WORK_UNITS, **dict(dies[FIRST].work_units)},
                store_queues={name: dict(queue) for name, queue in dies[FIRST].store_queues
                              if all(dict(d.store_queues).get(name) == queue for d in dies)},
                physical_mechanisms=dies[FIRST].physical_mechanisms,
            )
        return RecipeContext(
            None, {}, profile.names, self.resource_rates,
            profile.dies[FIRST].aic_count,
            pool_names=dict(profile.stage_pools),
            resource_units=dict(profile.units),
            compute_peaks=dict(profile.compute_peaks),
            pool_units=dict(profile.pool_units_by_name),
            pool_capacities=dict(profile.pool_capacities),
            record_mapping=self.cost.record_mapping,
            service_modes=dict(profile.service_modes),
            record_service=self.cost.record_service,
            work_units=dict(profile.work_units),
            store_queues={name: dict(queue)
                          for name, queue in profile.store_queues.items()},
            physical_mechanisms=profile.physical_mechanisms,
        )

    def _context(self, op: Any, replica: int, scope: str, *, recipe=None) -> RecipeContext:
        from .services import service_identity
        from .kernel_groups import declaration
        from .matrix_precision import execution_compute_peaks, resolve_matrix_precision
        from .collective_waves import startup_waves_recipe
        recipe = startup_waves_recipe(
            op, recipe, self.case.runtime.collective_startup_mode)
        dies = self.stage_dies[replica, op.stage]
        op = resolve_matrix_precision(
            op, devices=dies,
            policy=self.case.runtime.matrix_compute_policy, recipe=recipe)
        cost = self.cost.price(op, scope)
        template = self.stage_contexts[replica, op.stage]
        from .direct_peer import resolve_transport_options
        from .topology import TransportOptions
        transport_options = (resolve_transport_options(recipe, op)
                             if recipe is not None else TransportOptions())
        self._transport_options[(str(scope), op.op_id, replica)] = transport_options
        peaks = execution_compute_peaks(
            op, devices=dies, base_peaks=template.compute_peaks)
        meta = _metadata_dict(op)
        device = dies[0]
        units_key = _DEVICE_UNITS_CACHE.get(id(device))
        if units_key is None or units_key[0] is not device:
            units_key = (device, _freeze_key(dict(device.resource_units)))
            if len(_DEVICE_UNITS_CACHE) >= 64:
                _DEVICE_UNITS_CACHE.clear()
            _DEVICE_UNITS_CACHE[id(device)] = units_key
        sid_key = (
            replica, op.stage, device.rank, device.compute_units,
            units_key[SECOND],
            op.family, op.recipe, op.dtype, op.layout, scope,
            meta.get("kernel"), meta.get("kernel_name"),
            _freeze_key(meta.get("implementation")),
            _freeze_key(meta.get("service_axes")),
            bool(meta.get("hccl_selection")),
            # One metadata projection for both the prepared-program and the
            # service-identity caches: it is layer-normalized, so two layers
            # with the same declared kernel share the resolved identity.
            self._op_key(op),
        )
        sid = self._service_identity_cache.get(sid_key)
        if sid is None:
            sid = service_identity(self.case, op, scope, device=dies[0])
            if len(self._service_identity_cache) >= 1024:
                self._service_identity_cache.pop(next(iter(self._service_identity_cache)))
            self._service_identity_cache[sid_key] = sid
        declared = _declaration_of(op)
        context = replace(template, op=op, native_work=cost.work, compute_peaks=peaks,
            service_identity=sid,
            service_boundary="resource_phase" if declared else "device_kernel")
        bind_context = getattr(recipe, "bind_context", None)
        if bind_context is not None:
            context = bind_context(context)
        # A topology-declared direct mesh owns the physical plan for this
        # invocation.  Bind it before the ledger so execution, the isolated
        # row and the public accounting entry point consume one plan.
        algorithm = getattr(recipe, "algorithm", "ring")
        if (algorithm == "ring" and getattr(recipe, "mesh_capable", False)
                and self.topology.declares_direct_mesh(op, replica)):
            algorithm = "direct_mesh"
        if algorithm == "direct_mesh":
            plan = self.topology.joint_plan(
                op, {replica: cost.work.get("link", ZERO)},
                algorithm=algorithm, transport=transport_options)
            self.topology.bind_direct_mesh(op, replica, plan)
            plan_binding = _CollectivePlanBinding(algorithm, plan=plan)
        else:
            plan_binding = _CollectivePlanBinding(
                algorithm,
                factory=lambda: self.topology.joint_plan(
                    op, {replica: cost.work.get("link", ZERO)},
                    algorithm=algorithm, transport=transport_options))
        demands, frames, setup = {}, {}, 0.0
        if "peer_traffic" in dict(op.metadata) and not cost.work.get("link", 0):
            # A zero sender still participates and receives; validate its row.
            self.topology.accounting(op, replica, volume=cost.work.get("link", 0))
        for resource, amount in cost.work.items():
            if resource == "link":
                routed = self.topology.accounting(
                    op, replica, volume=amount, transport=transport_options)
                demands.update(routed.wire_by_resource)
                frames.update(routed.transaction_frames)
                # Envelope transport honors the declared round count. Explicit
                # collective programs own their hop arrivals and ignore setup.
                setup = routed.setup_ms
                continue
            if resource == "cube":
                portions = op.matrix_flops or ((op.dtype or self.case.model.compute_dtype, amount),)
                bound = context.bind(Work(matrix=dict(portions)))
            else:
                bound = context.bind(Work(resources={resource: amount}))
            for name, value in bound.items():
                demands[name] = demands.get(name, 0) + value
        diagnostic = (self.cost.lookup_diagnostic(op, scope)
                      if self.legacy_diagnostics else None)
        diagnostic_timing = (diagnostic or {}).get("timing")
        measured = (diagnostic_timing.duration_ms
                    if diagnostic_timing is not None else 0)
        physical = max((value / self.engine.resources[key].rate for key, value in demands.items()), default=0)
        # Context binding also serves independently constructed source phases;
        # native binding must not require an unrelated source-dispatch registration.
        context = replace(
            context, work=demands,
            minimum_ms=measured if self.legacy_diagnostics else 0.0,
            setup_ms=setup,
            service_identity={**dict(context.service_identity),
                              "transport_options": transport_options},
            transaction_frames=frames,
            collective_plan=plan_binding)
        # A declared service regime is a property of the declared kernel, not
        # of the diagnostic profile: ordinary production consults it too. A
        # case that declares no rule has nothing to resolve, so it is not
        # consulted at all (the legacy diagnostic path keeps its old access).
        # The observed MAC-as-work branch below stays diagnostic-only.
        response = (self.regimes.resolve(context, self.cost.record_regime)
                    if self.legacy_diagnostics or self.regimes.rules else None)
        if response is not None and response.mode == "joint_native":
            if getattr(recipe, "algorithm", None):
                raise ModelingError("joint_native_service_requires_single_local_kernel")
            demand = response.bind_joint(context)
            self.cost.record_joint_native(response, cost)
            declaration = {**response.execution, **response.diagnostics(),
                           "reference_program": type(recipe).__name__,
                           "reference_primitive_ms": (
                               measured if diagnostic_timing is not None else None)}
            return replace(context, work=demand, minimum_ms=0, setup_ms=0,
                           invocation_timing=None, joint_native_service=declaration)
        if (response is not None and response.mode == "mac_service"
                and self.legacy_diagnostics):
            matrix_role = str(response.execution.get("resource") or "cube")
            matrix_name = context.resource_names.get(matrix_role)
            if (not matrix_name or response.issued_work is None
                    or response.rate_flops_per_s is None
                    or not response.active_owners):
                raise ModelingError("mac_service_requires_bound_cube_work_and_rate")
            issued = float(response.issued_work)
            owners = int(response.active_owners)
            # ``cube`` work is physical time at peak (unit ``peak_ms``, rate 1).
            # The measured MAC time is per active owner; the fair-shared cube
            # resource yields that per-owner time while the tiled program
            # distributes it across the owner's tiles.  Do not multiply by the
            # owner count: that would serialize independent owners.
            mac_work_ms = float(response.duration_ms)
            peak_work_per_ms = float(context.compute_peaks.get(context.op.dtype, 0))
            physical_ms = (issued / (peak_work_per_ms / 1000.0)
                           if peak_work_per_ms > 0 else None)
            work = dict(context.work)
            work[matrix_name] = mac_work_ms
            declaration = {
                **response.execution, **response.diagnostics(),
                "stage": "mac", "resource_name": matrix_name,
                "issued_work": issued,
                "logical_work": float(cost.work.get("cube", 0.0)),
                "rate_flops_per_s": float(response.rate_flops_per_s),
                "active_owners": owners,
                "measured_mac_per_owner_ms": mac_work_ms,
                "mac_work_ms_total": mac_work_ms,
                "physical_mac_ms_at_peak": physical_ms,
                "reference_program": type(recipe).__name__,
            }
            program_recipe = None
            from .matrix import TiledMatmul
            if isinstance(recipe, TiledMatmul):
                source = TiledMatmul(
                    tile_m=int(response.execution["tile_m"]),
                    tile_n=int(response.execution["tile_n"]),
                    tile_k=int(response.execution["tile_k"]),
                    slots=int(response.execution["slots"]),
                )
                if source != recipe:
                    program_recipe = source
                    declaration = {**declaration,
                                   "source_selected_tiling": {
                                       "tile_m": source.tile_m, "tile_n": source.tile_n,
                                       "tile_k": source.tile_k, "slots": source.slots}}
            context = replace(context, work=work, mac_caps={},
                              mac_service=declaration,
                              program_recipe=program_recipe)
        self.conflicts += int(bool(measured and measured < physical))
        # A native peer plan owns one real invocation boundary in KernelGroups:
        # startup/minimum are applied once at its actual exit. Wrapping the
        # native program here would both duplicate that floor and hide its
        # launch handles from the shared cohort launcher.
        self._last_invocation_record = None
        selected = (None if getattr(recipe, "native", False)
                    else self.invocations.select(context, recipe, devices=dies))
        if selected is not None:
            timing, record = selected
            self._last_invocation_record = record
            self.cost.record_invocation(record)
            context = replace(context, invocation_timing=timing)
        if response is not None:
            if response.mode == "mac_service":
                context = replace(context, service_regime=response.diagnostics())
            else:
                # The response is an inclusive completion floor: the general
                # physical program keeps its own work and busy time, and the
                # declared service duration can only raise the call's end.
                existing = float(context.minimum_ms or 0.0)
                resp_prov = getattr(response, "provenance", None)
                provenance = (dict(resp_prov or {})
                              if float(response.duration_ms) >= existing
                              else dict(getattr(context, "minimum_provenance", {}) or {}))
                context = replace(
                    context,
                    minimum_ms=max(existing, response.duration_ms),
                    minimum_provenance=provenance,
                    service_regime=response.diagnostics())
        return context

    def _add_lazy(self, work, replica, release_ms, *, scope, schedule,
                  prepared, input_ready, physical_ops, prefix,
                  external_inputs, emitted_tokens, memory_bindings,
                  state_accesses, verified_dag=None):
        """Materialize only the active dependency frontier.

        A real op completion resolves one promise; only successors whose real
        predecessors have all completed are activated. Host launches are
        materialized a bounded lookahead ahead of the device frontier, so the
        genuine host FIFO can lead device without registering every dormant
        op's ready/begin/program tasks. Complex paths (collectives, kernel
        groups, graph capture, offload/state copies) keep the detailed loop.
        """
        engine = self.engine
        assert self.offload_policy is None
        n_ops = len(physical_ops)
        if verified_dag is not None:
            (order, succ_idx, init_unmet_t, dep_specs,
             root_idx, sink_idx, fut_indices) = verified_dag
        else:
            dag_sig = tuple((op.op_id, op.deps, op.family == "token_commit")
                            for op in physical_ops)
            prev_dag = self._prev_lazy_dag
            if prev_dag is not None and prev_dag[0] == dag_sig:
                (order, succ_idx, init_unmet_t, dep_specs,
                 root_idx, sink_idx, fut_indices) = prev_dag[1]
            else:
                order = {op.op_id: index for index, op in enumerate(physical_ops)}
                succ_lists: list[list[int]] = [[] for _ in range(n_ops)]
                init_unmet = [0] * n_ops
                dep_specs_list: list[Any] = [None] * n_ops
                for idx, op in enumerate(physical_ops):
                    raw_deps = op.deps
                    if not raw_deps:
                        dep_specs_list[idx] = ()
                        continue
                    if len(raw_deps) == 1:
                        d0 = raw_deps[0]
                        j0 = order.get(d0)
                        if j0 is not None:
                            init_unmet[idx] = 1
                            succ_lists[j0].append(idx)
                            dep_specs_list[idx] = j0
                            continue
                    int_d = []
                    all_internal = True
                    for dep in raw_deps:
                        j = order.get(dep)
                        if j is not None:
                            int_d.append(j)
                            succ_lists[j].append(idx)
                        else:
                            all_internal = False
                    init_unmet[idx] = len(int_d)
                    dep_specs_list[idx] = tuple(int_d) if all_internal else None
                root_idx = tuple(i for i in range(n_ops) if not init_unmet[i])
                if n_ops and not root_idx:
                    raise ModelingError("execution_dependency_cycle")
                reachable = [False] * n_ops
                reach_count = 0
                stack = list(root_idx)
                while stack:
                    cur = stack.pop()
                    if reachable[cur]:
                        continue
                    reachable[cur] = True
                    reach_count += 1
                    stack.extend(succ_lists[cur])
                if reach_count != n_ops:
                    raise ModelingError("execution_dependency_cycle")
                sink_idx = tuple(i for i in range(n_ops) if not succ_lists[i])
                fut_indices = tuple(
                    i for i in range(n_ops)
                    if not succ_lists[i] or physical_ops[i].family == "token_commit"
                )
                succ_idx = tuple(tuple(s) for s in succ_lists)
                init_unmet_t = tuple(init_unmet)
                dep_specs = tuple(dep_specs_list)
                self._prev_lazy_dag = (
                    dag_sig,
                    (order, succ_idx, init_unmet_t, dep_specs,
                     root_idx, sink_idx, fut_indices),
                )
        unmet = list(init_unmet_t)
        op_events: list[Event | None] = [None] * n_ops
        op_futures: list[Event | None] = [None] * n_ops
        for fi in fut_indices:
            fut = engine.future(f"{prefix}:{physical_ops[fi].op_id}:lazy_complete")
            op_futures[fi] = fut
            op_events[fi] = fut
        ext_map = dict(external_inputs) if external_inputs else None
        declared_commits = {}
        for fi in fut_indices:
            fop = physical_ops[fi]
            if fop.family == "token_commit":
                fut = op_futures[fi]
                for request in fop.requests:
                    declared_commits[request] = fut
        host_info: list[Any] = [None] * n_ops
        host_cursor = 0
        host_submits: list[tuple[int, int, Event]] = []
        step_holder: dict[str, Any] = {}
        owner = f"replica:{replica}"
        from .program import NativeProgram
        compact_exec = (
            not self.legacy_diagnostics
            and (not engine.trace.maxlen or n_ops > engine.trace.maxlen)
        )
        compact_submit = compact_exec and n_ops > 32
        self._compact_step = compact_exec
        if hasattr(engine, "_compact_step"):
            engine._compact_step = compact_exec
        lazy_fast = (
            not self.legacy_diagnostics
            and not self.regimes.rules
            and not Execution.schedule_replay
        )
        stage_rates_map = (
            {stg: tuple(engine.resources[rn].rate for rn in s_names.values())
             for (rep, stg), s_names in self.stage_names.items()
             if rep == replica}
            if lazy_fast else {}
        )
        include_sub = self.include_submission
        stage_launch_ms = (
            {stg: self.submission.launch_ms(dies)
             for (rep, stg), dies in self.stage_dies.items()
             if rep == replica}
            if include_sub else {}
        )
        stage_host_prep = (
            {stg: engine._prepare(
                ":submit", {self.stage_names[(replica, stg)]["host"]: dur})
             for stg, dur in stage_launch_ms.items()}
            if include_sub else {}
        )
        stage_zero_host_prep = (
            {stg: engine._prepare(
                ":submit", {self.stage_names[(replica, stg)]["host"]: 0.0})
             for stg in stage_launch_ms}
            if (include_sub and compact_submit) else {}
        )
        has_fast_mem = hasattr(engine, "_memory_context_fast")
        default_ledger = (self.memory if self.memory is not None
                          else getattr(engine, "memory", None))
        if has_fast_mem and default_ledger is not None:
            engine._memory_ledgers[id(default_ledger)] = default_ledger
        stage_mem_info: dict[int, tuple[str, dict | None, tuple | None]] = {}
        for (rep, stg) in self.stage_names:
            if rep == replica:
                phbm = self.stage_pools.get((rep, stg), {}).get("hbm", "hbm")
                if has_fast_mem:
                    pfroz = frozenset((str(phbm),))
                    rep_i = int(replica)
                    stg_i = int(stg)
                    mctx = {
                        "memory": default_ledger, "pools": pfroz,
                        "replica": rep_i, "stage": stg_i, "bindings": {},
                    }
                    mfast = (default_ledger, pfroz, None, rep_i, stg_i)
                    stage_mem_info[stg] = (phbm, mctx, mfast)
                else:
                    stage_mem_info[stg] = (phbm, None, None)
        shared_zero_submit: dict[tuple[int, int], tuple[Event, Event]] = {}
        prefix_colon = prefix + ":"
        host_lookahead = self.host_lookahead
        observer = self.on_operator
        recipes_id = id(self.recipes)
        scope_str = str(scope)
        stage_id_caches = (
            {
                stg: self._lazy_stage_id_caches.setdefault(
                    (replica, scope, compact_exec, recipes_id, lrates), {}
                )
                for stg, lrates in stage_rates_map.items()
            }
            if lazy_fast else {}
        )
        step_env = (
            (replica, scope, compact_exec, recipes_id,
             tuple(sorted(stage_rates_map.items())))
            if lazy_fast else None
        )
        prev_le = self._prev_lazy_entries
        if (step_env is not None and prev_le is not None
                and prev_le[0] == step_env and len(prev_le[1]) == n_ops):
            prev_l_ops = prev_le[1]
            prev_l_entries = prev_le[2]
        else:
            prev_l_ops = None
            prev_l_entries = None
        cur_l_entries: list[Any] | None = [None] * n_ops if lazy_fast else None
        if step_env is not None:
            self._prev_lazy_entries = (step_env, physical_ops, cur_l_entries)
        lazy_cache = self._lazy_op_cache
        cost_counts = self.cost.counts
        cost_invs = self.cost.invocations
        trans_opts = self._transport_options
        streams = self.streams
        recipes_used = self.recipes_used
        exec_work = self.work
        replay_counts = self.replay_counts

        seen_host_stages: set[tuple[int, int]] = set()

        def update_host_submits():
            step = step_holder.get("step")
            if step is not None:
                step.host_submits = tuple(host_submits)

        def _submit_host_launch(stage_key, name):
            stg_id = stage_key[1]
            duration = stage_launch_ms[stg_id]
            if compact_submit and duration == 0.0:
                shared = shared_zero_submit.get(stage_key)
                if shared is None:
                    prev_h = self.hosts.get(stage_key)
                    host_deps = (prepared, prev_h) if prev_h is not None else prepared
                    launch, start = engine._submit_host_launch(
                        stage_zero_host_prep[stg_id], name + ":submit",
                        name + ":host_submit", host_deps, owner)
                    self.hosts[stage_key] = launch
                    seen_host_stages.add(stage_key)
                    host_submits.append((replica, stg_id, start))
                    shared_zero_submit[stage_key] = (launch, start)
                else:
                    launch, start = shared
                self.submission_counts["eager_invocations"] += 1
                return launch, start
            prev_h = self.hosts.get(stage_key)
            if stage_key in seen_host_stages:
                host_deps = prev_h
            else:
                seen_host_stages.add(stage_key)
                host_deps = (prepared, prev_h) if prev_h is not None else prepared
            launch, start = engine._submit_host_launch(
                stage_host_prep[stg_id], name + ":submit",
                name + ":host_submit", host_deps, owner)
            self.hosts[stage_key] = launch
            self.submission_counts["eager_invocations"] += 1
            host_submits.append((replica, stg_id, start))
            return launch, start

        def materialize_one(idx: int, op):
            nonlocal host_cursor
            entry = None
            lazy_key = None
            stg_id_cache = None
            same_step_op = False
            if lazy_fast:
                if (prev_l_ops is not None and prev_l_ops[idx] is op
                        and (pe := prev_l_entries[idx]) is not None):
                    entry = pe
                    same_step_op = True
                else:
                    stg_id_cache = stage_id_caches[op.stage]
                    hit = stg_id_cache.get(id(op))
                    if hit is not None and hit[FIRST] is op:
                        entry = hit[1]
                    else:
                        live_rates = stage_rates_map[op.stage]
                        lazy_key = (
                            replica, scope, compact_exec, recipes_id, live_rates,
                            op.stage, op.family, op.recipe, op.phase, op.dtype, op.layout,
                            op.semantic_shape, op.local_shape, op.padded_shape, op.launch_shape,
                            op.flops, op.matrix_flops, op.mte2_bytes, op.mte3_bytes,
                            op.vector_ops, op.hbm_bytes, op.link_bytes, op.link_scope,
                            op.source_boundary, op.returns, self._op_key(op),
                        )
                        entry = lazy_cache.get(lazy_key)
                        if entry is not None and entry is not False:
                            if len(stg_id_cache) >= 16384:
                                stg_id_cache.clear()
                            stg_id_cache[id(op)] = (op, entry)
                if (entry is not None and entry is not False
                        and entry[SECOND] is not False):
                    if cur_l_entries is not None:
                        cur_l_entries[idx] = entry
                    (res_op_tmpl, _mdiff, _rmem, _rec, _ctx_tmpl,
                     submit, transport_options, inv_cell, _rname, _oname,
                     _wdeltas) = entry[FIRST]
                    cost_counts["analytic_work"] += 1
                    if not same_step_op:
                        trans_opts[(scope_str, op.op_id, replica)] = transport_options
                    if inv_cell is not None:
                        if inv_cell[2] is cost_invs and inv_cell[3] is not None:
                            inv_cell[3]["invocations"] += 1
                        elif inv_cell[FIRST] in cost_invs:
                            li = cost_invs[inv_cell[0]]
                            inv_cell[2] = cost_invs
                            inv_cell[3] = li
                            li["invocations"] += 1
                        else:
                            self.cost.record_invocation(inv_cell[SECOND])
                            inv_cell[2] = cost_invs
                            inv_cell[3] = cost_invs.get(inv_cell[0])
                    res_stg = res_op_tmpl.stage
                    name = prefix_colon + op.op_id
                    launch = start = None
                    if submit and include_sub:
                        stage_key = (replica, res_stg)
                        launch, start = _submit_host_launch(stage_key, name)
                    host_info[idx] = (entry, op, name, launch, start, None)
                    host_cursor += 1
                    return

            res_op, recipe = self.prepare(op, replica)
            stage_key = replica, res_op.stage
            name = prefix_colon + res_op.op_id
            context = self._context(res_op, replica, scope, recipe=recipe)
            submit = bool(context.work or getattr(recipe, "algorithm", None))
            launch = start = None
            if submit and include_sub:
                launch, start = _submit_host_launch(stage_key, name)
            fast_created = None
            if lazy_fast and lazy_key is not None and entry is None:
                orig_meta = dict(op.metadata)
                res_meta = dict(res_op.metadata)
                meta_diff = _meta_diff(orig_meta, res_meta) if orig_meta != res_meta else None
                raw_mm = res_meta.get("memory_bindings", res_meta.get("memory_binding"))
                raw_meta_mem = (
                    {str(mn): dict(mv) for mn, mv in raw_mm.items()
                     if isinstance(mv, Mapping)}
                    if isinstance(raw_mm, Mapping) else None
                )
                transport_options = trans_opts[(scope_str, res_op.op_id, replica)]
                inv_record = self._last_invocation_record
                inv_cell = (
                    [inv_record["id"], inv_record, cost_invs,
                     cost_invs.get(inv_record["id"])]
                    if inv_record is not None else None
                )
                recipe_name = res_op.recipe or type(recipe).__name__
                obs_prog_name = (
                    "JointNativeService"
                    if context.joint_native_service is not None
                    else type(recipe).__name__
                )
                work_deltas = tuple(
                    (k, v) for k in exec_work
                    if (v := float(getattr(res_op, k))) != 0.0
                )
                prep_bundle = (
                    res_op, meta_diff, raw_meta_mem, recipe, context,
                    submit, transport_options, inv_cell, recipe_name, obs_prog_name, work_deltas,
                )
                fast_created = [prep_bundle, None]
                if len(lazy_cache) >= 4096:
                    lazy_cache.pop(next(iter(lazy_cache)))
                lazy_cache[lazy_key] = fast_created
                if stg_id_cache is not None:
                    if len(stg_id_cache) >= 16384:
                        stg_id_cache.clear()
                    stg_id_cache[id(op)] = (op, fast_created)
                if cur_l_entries is not None:
                    cur_l_entries[idx] = fast_created
            if fast_created is not None:
                host_info[idx] = (fast_created, op, name, launch, start, None)
            else:
                host_info[idx] = (None, op, name, launch, start, (res_op, recipe, context))
            host_cursor += 1

        def materialize_through(limit_idx: int):
            nonlocal host_cursor
            end = n_ops - 1 if limit_idx >= n_ops else limit_idx
            if host_cursor > end:
                return
            prev_len = len(host_submits)
            while host_cursor <= end:
                cur = host_cursor
                if host_info[cur] is None:
                    materialize_one(cur, physical_ops[cur])
                else:
                    host_cursor = cur + 1
            if step_holder and len(host_submits) != prev_len and host_cursor >= n_ops:
                update_host_submits()

        def _run_cached_bundle(op, name, ready_ev, fast_entry, res_op_tmpl,
                               meta_diff, context_tmpl):
            prog_bundle = fast_entry[1]
            if prog_bundle is not None:
                (program, mapped_entries, svc_calls,
                 is_native_prog, has_handles, transport) = prog_bundle
                replay_counts["real"] += 1
                if mapped_entries and context_tmpl.record_mapping is not None:
                    mappings = self.cost.mappings
                    for m_item in mapped_entries:
                        if m_item[2] is mappings and m_item[3] is not None:
                            m_item[3]["invocations"] += 1
                        elif m_item[0] in mappings:
                            lm = mappings[m_item[0]]
                            m_item[2] = mappings
                            m_item[3] = lm
                            lm["invocations"] += 1
                        else:
                            context_tmpl.record_mapping(m_item[1])
                            m_item[2] = mappings
                            m_item[3] = mappings.get(m_item[0])
                if svc_calls and context_tmpl.record_service is not None:
                    for sargs, skwargs in svc_calls:
                        context_tmpl.record_service(*sargs, **skwargs)
                if has_handles and op.op_id not in self.native_group_launches:
                    meta = op.metadata
                    if meta_diff is not None:
                        patched = _meta_patch(dict(op.metadata), meta_diff)
                        meta = tuple(sorted(patched.items()))
                    cur_res_op = (
                        res_op_tmpl if res_op_tmpl.op_id == op.op_id
                        else replace(res_op_tmpl, op_id=op.op_id,
                                     layer=op.layer, metadata=meta)
                    )
                    self._record_single_native(
                        replace(context_tmpl, op=cur_res_op),
                        program, replica, name, transport=transport)
                if is_native_prog:
                    if transport is None:
                        return program.program.start(engine, name, ready_ev)
                    return program.start(engine, name, ready_ev, transport=transport)
                return program.start(engine, name, ready_ev, owner=owner)

            recipe = fast_entry[0][3]
            meta = op.metadata
            if meta_diff is not None:
                patched = _meta_patch(dict(op.metadata), meta_diff)
                meta = tuple(sorted(patched.items()))
            cur_res_op = (
                res_op_tmpl if res_op_tmpl.op_id == op.op_id
                else replace(
                    res_op_tmpl, op_id=op.op_id, layer=op.layer,
                    deps=op.deps, requests=op.requests,
                    stream=op.stream, notes=op.notes, metadata=meta)
            )
            ctx = replace(context_tmpl, op=cur_res_op)
            mapped_list = []
            svc_list = []
            orig_rec = ctx.record_mapping
            orig_svc = ctx.record_service

            def cap_map(rec):
                mapped_list.append(deepcopy(rec))
                if orig_rec is not None:
                    orig_rec(rec)

            def cap_svc(*args, **kwargs):
                svc_list.append((deepcopy(args), deepcopy(kwargs)))
                if orig_svc is not None:
                    orig_svc(*args, **kwargs)

            run_ctx = replace(
                ctx,
                record_mapping=cap_map if orig_rec is not None else None,
                record_service=cap_svc if orig_svc is not None else None,
            )
            replay_counts["real"] += 1
            program = self.lower_program(run_ctx, recipe)
            program = self._apply_resource_service_program(ctx, program)
            if compact_exec:
                from .program import coalesce_execution_program
                program = coalesce_execution_program(program)
            is_native_prog = isinstance(program, NativeProgram)
            has_handles = is_native_prog and hasattr(
                program.program, "start_with_handles")
            transport = (
                self.native_transport(cur_res_op, replica)
                if getattr(recipe, "native", False) else None
            )
            mappings_init = self.cost.mappings
            mapped_entries = tuple(
                [mk := json.dumps(m, sort_keys=True, separators=(",", ":")),
                 m, mappings_init, mappings_init.get(mk)]
                for m in mapped_list
            )
            fast_entry[1] = (
                program, mapped_entries, tuple(svc_list),
                is_native_prog, has_handles, transport,
            )
            if has_handles and cur_res_op.op_id not in self.native_group_launches:
                self._record_single_native(
                    ctx, program, replica, name, transport=transport)
            if is_native_prog:
                return program.start(engine, name, ready_ev, transport=transport)
            return program.start(engine, name, ready_ev, owner=owner)

        def _collect_begin_after(idx: int, op, launch, tail):
            dspec = dep_specs[idx]
            if dspec.__class__ is int:
                d_ev = op_events[dspec]
                if d_ev is not None:
                    if launch is not None and tail is not None:
                        return (input_ready, d_ev, launch, tail)
                    if launch is not None:
                        return (input_ready, d_ev, launch)
                    if tail is not None:
                        return (input_ready, d_ev, tail)
                    return (input_ready, d_ev)
                if launch is not None and tail is not None:
                    return (input_ready, launch, tail)
                if launch is not None:
                    return (input_ready, launch)
                if tail is not None:
                    return (input_ready, tail)
                return (input_ready,)
            deps = [input_ready]
            if dspec is not None:
                for j in dspec:
                    ev = op_events[j]
                    if ev is not None:
                        deps.append(ev)
            else:
                for dep in op.deps:
                    j = order.get(dep)
                    if j is not None:
                        ev = op_events[j]
                    elif ext_map is not None:
                        ev = ext_map.get(dep)
                    else:
                        ev = None
                    if ev is not None:
                        deps.append(ev)
            if launch is not None:
                deps.append(launch)
            if tail is not None:
                deps.append(tail)
            return tuple(deps) or (schedule,)

        def activate(idx: int):
            info = host_info[idx]
            if info is None:
                materialize_through(idx)
                info = host_info[idx]
            fast_entry, orig_op, name, launch, _start, slow_pack = info
            op = physical_ops[idx]
            if fast_entry is not None and fast_entry[1] is False:
                res_op, recipe = self.prepare(op, replica)
                context = self._context(res_op, replica, scope, recipe=recipe)
                slow_pack = (res_op, recipe, context)
                fast_entry = None
            if fast_entry is not None:
                (res_op_tmpl, meta_diff, raw_meta_mem, _recipe, context_tmpl,
                 _submit, _topt, _inv, recipe_name, obs_prog_name, work_deltas) = fast_entry[0]
                res_stage = res_op_tmpl.stage
                stream_key = (replica, res_stage,
                              op.stream or res_op_tmpl.stream or "main")
                tail = streams.get(stream_key)
                begin_after = _collect_begin_after(idx, op, launch, tail)
                mem_info = stage_mem_info.get(res_stage)
                if mem_info is not None:
                    physical_hbm, mctx_fast_dict, mfast_tuple = mem_info
                else:
                    physical_hbm = self.stage_pools.get(
                        (replica, res_stage), {}).get("hbm", "hbm")
                    mctx_fast_dict = mfast_tuple = None
                if memory_bindings is None:
                    mem_bind = (
                        {mn: dict(mv) for mn, mv in raw_meta_mem.items()}
                        if raw_meta_mem is not None else None
                    )
                else:
                    meta = op.metadata
                    if meta_diff is not None:
                        patched = _meta_patch(dict(op.metadata), meta_diff)
                        meta = tuple(sorted(patched.items()))
                    op_for_mem = replace(res_op_tmpl, op_id=op.op_id, metadata=meta)
                    mem_bind = memory_bindings_for(op_for_mem, memory_bindings)

                now = engine.now
                sync_ev = None
                all_done = True
                for ev in begin_after:
                    ev_t = ev.time
                    if ev_t is None or ev_t > now:
                        all_done = False
                        sync_ev = None
                        break
                    if ev_t == now:
                        sync_ev = ev
                pending_evs = None
                if all_done and sync_ev is None and compact_exec and begin_after:
                    sync_ev = engine._instant_done_event(
                        name + ":begin" if observer is None else name + ":ready")
                elif not all_done and compact_exec and observer is None:
                    pending_evs = [
                        ev for ev in begin_after
                        if ev.time is None or ev.time > now
                    ]
                    pb = fast_entry[1]
                    if pb is not None and not pb[4] and pb[5] is None:
                        prog = pb[0].program if pb[3] else pb[0]
                        if prog.__class__ is KernelProgram and prog.minimum_ms <= 0.0:
                            comp_c = getattr(engine, "_compiled_programs", None)
                            chit = comp_c.get(id(prog)) if comp_c is not None else None
                            if (chit is not None and chit[0] is prog
                                    and chit[1] is not False
                                    and not chit[1][3]):
                                if len(pending_evs) == 1:
                                    sync_ev = pending_evs[0]
                                elif len(pending_evs) > 1 and chit[1][0]:
                                    sync_ev = tuple(pending_evs)
                if sync_ev is not None:
                    if mem_bind is None and mfast_tuple is not None:
                        prev_mc = engine._memory_context
                        prev_mf = engine._memory_context_fast
                        engine._memory_context = mctx_fast_dict
                        engine._memory_context_fast = mfast_tuple
                        try:
                            complete = _run_cached_bundle(
                                op, name, sync_ev, fast_entry,
                                res_op_tmpl, meta_diff, context_tmpl)
                        finally:
                            engine._memory_context = prev_mc
                            engine._memory_context_fast = prev_mf
                    else:
                        with engine.memory_scope(
                                self.memory, replica=replica, stage=int(res_stage),
                                pools=(physical_hbm,), bindings=mem_bind):
                            complete = _run_cached_bundle(
                                op, name, sync_ev, fast_entry,
                                res_op_tmpl, meta_diff, context_tmpl)
                    streams[stream_key] = complete
                    op_events[idx] = complete
                    recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
                    if observer is not None:
                        observer(orig_op, replica, obs_prog_name, sync_ev, complete)
                    for key, delta in work_deltas:
                        exec_work[key] += delta
                    if host_cursor < n_ops:
                        materialize_through(idx + host_lookahead)
                    if complete.time is not None:
                        if op_futures[idx] is not None or succ_idx[idx]:
                            completed(idx)
                    elif op_futures[idx] is not None or succ_idx[idx]:
                        t_comp = engine._tasks.get(complete.sequence)
                        cb = lambda _ev, _i=idx: completed(_i)
                        if t_comp is not None and t_comp.callback is None:
                            t_comp.callback = cb
                        else:
                            engine.on_complete(complete, cb)
                    return

                complete = engine.future(name + ":program_complete")

                def async_begin(boundary_ev, stg=int(res_stage), phbm=physical_hbm,
                                mb=mem_bind, mcd=mctx_fast_dict, mft=mfast_tuple,
                                comp=complete, _i=idx, _op=op, _nm=name,
                                _fe=fast_entry, _rt=res_op_tmpl, _md=meta_diff,
                                _ct=context_tmpl):
                    if mb is None and mft is not None:
                        prev_mc = engine._memory_context
                        prev_mf = engine._memory_context_fast
                        engine._memory_context = mcd
                        engine._memory_context_fast = mft
                        try:
                            finish = _run_cached_bundle(
                                _op, _nm, boundary_ev, _fe, _rt, _md, _ct)
                        finally:
                            engine._memory_context = prev_mc
                            engine._memory_context_fast = prev_mf
                    else:
                        with engine.memory_scope(
                                self.memory, replica=replica, stage=stg,
                                pools=(phbm,), bindings=mb):
                            finish = _run_cached_bundle(
                                _op, _nm, boundary_ev, _fe, _rt, _md, _ct)

                    def done_cb(_ev=None):
                        engine.resolve(comp)
                        completed(_i)

                    engine.on_complete(finish, done_cb)

                if compact_exec and pending_evs is None:
                    pending_evs = [
                        ev for ev in begin_after
                        if ev.time is None or ev.time > now
                    ]
                if pending_evs is not None and len(pending_evs) == 1:
                    ready_marker = pending_evs[0]
                    engine.on_complete(ready_marker, async_begin)
                else:
                    ready_marker = engine.submit(
                        name + ":begin" if observer is None else name + ":ready",
                        after=begin_after, owner=owner,
                        on_complete=async_begin)
                streams[stream_key] = complete
                op_events[idx] = complete
                recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
                if observer is not None:
                    observer(orig_op, replica, obs_prog_name, ready_marker, complete)
                for key, delta in work_deltas:
                    exec_work[key] += delta
                if host_cursor < n_ops:
                    materialize_through(idx + host_lookahead)
                return

            res_op, recipe, context = slow_pack
            stage_key = replica, res_op.stage
            stream_key = replica, res_op.stage, res_op.stream or "main"
            tail = streams.get(stream_key)
            begin_after = _collect_begin_after(idx, op, launch, tail)
            physical_hbm = self.stage_pools.get(stage_key, {}).get(
                "hbm", "hbm")
            if observer is None:
                ready = None
                complete = self._start_program(
                    context, recipe, name, None, owner=owner, replica=replica,
                    memory_ledger=self.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(res_op, memory_bindings),
                    on_resolve=lambda _i=idx: completed(_i),
                    begin_after=begin_after)
            else:
                ready = engine.submit(
                    name + ":ready", after=begin_after, owner=owner)
                complete = self._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica,
                    memory_ledger=self.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(res_op, memory_bindings),
                    on_resolve=lambda _i=idx: completed(_i))
            streams[stream_key] = complete
            op_events[idx] = complete
            recipe_name = res_op.recipe or type(recipe).__name__
            recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
            if observer is not None:
                observer(
                    orig_op, replica,
                    ("JointNativeService" if context.joint_native_service
                     is not None else type(recipe).__name__),
                    ready, complete)
            for key in exec_work:
                exec_work[key] += float(getattr(res_op, key))
            materialize_through(idx + host_lookahead)

        activating = False
        pending_activations = deque()

        def drain_activations():
            nonlocal activating
            if activating:
                return
            activating = True
            try:
                while pending_activations:
                    activate(pending_activations.popleft())
            finally:
                activating = False

        def completed(idx: int):
            nonlocal activating
            fut = op_futures[idx]
            if fut is not None:
                engine.resolve(fut)
            succs = succ_idx[idx]
            n_s = len(succs)
            if n_s == 1:
                s0 = succs[0]
                u0 = unmet[s0] - 1
                unmet[s0] = u0
                if not u0:
                    if activating:
                        pending_activations.append(s0)
                    else:
                        activating = True
                        try:
                            activate(s0)
                            while pending_activations:
                                activate(pending_activations.popleft())
                        finally:
                            activating = False
                return
            if not n_s:
                return
            ready_ops = None
            for s in succs:
                us = unmet[s] - 1
                unmet[s] = us
                if not us:
                    if ready_ops is None:
                        ready_ops = [s]
                    else:
                        ready_ops.append(s)
            if ready_ops is not None:
                if len(ready_ops) > 1:
                    ready_ops.sort()
                pending_activations.extend(ready_ops)
                drain_activations()

        materialize_through(
            (root_idx[-1] + host_lookahead) if root_idx else host_lookahead)
        if root_idx:
            pending_activations.extend(root_idx)
            drain_activations()
        device_complete = engine.submit(
            prefix + ":device_complete", owner=owner,
            after=tuple(op_futures[si] for si in sink_idx) or (input_ready,))
        completion = engine.submit(
            prefix + ":complete", after=(device_complete,),
            latency=(self.submission.completion_ms
                     if self.include_submission else 0),
            owner=owner)
        fallback_commits = {
            view.request_id: completion for view in work.plan.requests
            if view.request_id not in declared_commits}
        commits = {**fallback_commits, **declared_commits}
        host_results, clients, host_complete = self.serving_io.output(
            self, prefix, replica, work, commits, completion,
            emitted_tokens if emitted_tokens is not None else {
                view.request_id: int(view.draft_advanced_tokens or 1)
                for view in work.plan.requests if view.samples})
        if self.serving_io.enabled:
            completion = engine.submit(
                prefix + ":step_ready", after=(completion, host_complete),
                owner=owner)
        from .offload import release_offload_access
        # Zero-work bookkeeping rows: their completion hook is virtual-time
        # independent, so the kernel may mirror them as an exact boundary.
        offload_release = engine.submit(
            prefix + ":offload_release", after=(device_complete,),
            owner=owner, native_hook=True,
            on_complete=lambda _event: release_offload_access(self, work))
        step = ScheduledIteration(
            replica, work.plan.iteration, completion, release_ms,
            work.mode, work.valid_tokens, work.padded_tokens,
            work.graph_bucket, scope, len(physical_ops), commits,
            device_complete, host_results, clients,
            work.implementation_selections,
            tuple(host_submits),
            dict(declared_commits), dict(fallback_commits),
            offload_release)
        step_holder["step"] = step
        update_host_submits()

        def _record_step(_):
            if hasattr(engine, "_compact_step"):
                engine._compact_step = False
            update_host_submits()
            self.step_summary.record(step)

        engine.submit(prefix + ":step_metrics", after=(completion,),
                      on_complete=_record_step, native_hook=True)
        return step

    def _lower_local_tp_work(self, work: IterationWork, replica: int) -> IterationWork:
        """Apply the declared node-local dense placement once per graph."""
        parallel = self.case.runtime.parallel
        if max(int(parallel.o_proj_tp), int(parallel.linear_proj_tp),
               int(parallel.lm_head_tp)) <= ONE:
            return work
        from .local_projection_tp import lower_local_projection_tp

        cohort_views = self._cohort_views
        if cohort_views is None:
            parallel = self.case.runtime.parallel
            local_width = max(int(parallel.tp), int(parallel.o_proj_tp),
                              int(parallel.linear_proj_tp),
                              int(parallel.lm_head_tp))
            if (int(parallel.tp) == ONE and local_width > ONE
                    and (work.plan.requests or work.mode != "graph")):
                # A homogeneous single-replica evaluation still represents the
                # complete node-local TP cohort.  Populate every peer's actual
                # request rows so the rewritten GEMM receives the full node
                # token count with 1/local_width features, not 1/local_width
                # of the tokens.
                from .local_tp_groups import local_tp_replica_cohorts
                cohorts = local_tp_replica_cohorts(
                    self.case.machine, parallel)
                peers = next((group for group in cohorts if replica in group),
                             (replica,))
                cohort_views = {
                    int(rank): work.plan.requests
                    for peer in peers
                    for rank in self.topology.ranks(peer, ZERO)
                }
            else:
                cohort_views = (
                    {int(rank): work.plan.requests
                     for rank in self.topology.ranks(replica, ZERO)}
                    if work.plan.requests or work.mode != "graph"
                    else {}
                )
        ops = lower_local_projection_tp(
            work.ops, model=self.case.model, runtime=self.case.runtime,
            machine=self.case.machine, replica=replica,
            cohort_views=cohort_views, scope="target")
        return work if ops is work.ops else replace(work, ops=ops)

    def add(self, work: IterationWork, replica: int, release_ms: float, *,
            external_inputs=None, emitted_tokens=None,
            memory_bindings=None, state_accesses=None) -> ScheduledIteration:
        if self._collectives is None:
            if _NO_REV_OPS_CACHE.get(id(work.ops)) is not work.ops:
                has_rev = False
                for op in work.ops:
                    if op.metadata and meta_scan_flags(op.metadata)[1]:
                        has_rev = True
                        break
                if has_rev:
                    work = replace(work, ops=bind_cohort_traffic(
                        self.case, {replica: work.ops})[replica])
                else:
                    mark_no_rev_collective(work.ops)
        work = self._lower_local_tp_work(work, replica)
        engine = self.engine
        prefix = f"{self.namespace}r{replica}:i{work.plan.iteration}"
        ids: dict[str, Event] = dict(external_inputs or {})
        declared_commits: dict[str, Event] = {}
        host_submits: list[tuple[int, int, Event]] = []
        scope = work.replay_scope or ("full" if work.graph_bucket is not None else "none")
        if self.include_submission and self.case.runtime.async_scheduling and replica in self.prepared:
            schedule = self.prepared.pop(replica)
        else:
            scheduler_ms = self.submission.scheduler_ms if self.include_submission else 0
            schedule = engine.submit(prefix + ":schedule", {f"{self.namespace}scheduler@r{replica}": scheduler_ms},
                                     release=release_ms, owner=f"replica:{replica}")
        self.schedules[replica] = schedule
        prepared, input_ready = self.serving_io.input(self, prefix, replica, schedule, work)
        graph_launch: dict[object, tuple[Event, Event]] = {}
        layer_entries: dict[int, Event] = {}
        expert_layer_events: dict[tuple[int, int], Event | None] = {}
        last_expert_done: dict[int, Event] = {}
        kv_layer_events: dict[tuple[int, int], list[Event]] = {}
        last_kv_done: dict[int, Event] = {}
        kv_writeback_layers: set[tuple[int, int]] = set()
        kv_writeback_events: list[Event] = []
        all_bound_accesses: list = []
        owned_offload: list[Event] = []
        pending_draft: dict[int, list] = {}
        last_draft_done: dict[int, Event] = {}
        physical_ops = self.recipes.expand(work.ops)
        external = dict(external_inputs or {})

        safe_cache = self._lazy_safe_id_cache

        def _lazy_safe(op):
            if not op.metadata:
                return True
            hit = safe_cache.get(id(op))
            if hit is not None and hit[FIRST] is op:
                return hit[SECOND]
            meta = _metadata_dict(op)
            ok = (_declaration_of(op) is None
                  and meta.get("group_axis") not in {"ep", "local_tp"}
                  and not meta.get("peer_traffic")
                  and not meta.get("mc2_phase")
                  and not meta.get("reverse_of_collective"))
            if len(safe_cache) >= 16384:
                safe_cache.clear()
            safe_cache[id(op)] = (op, ok)
            return ok

        if (self.lazy_frontier and physical_ops
                and scope == "none" and self._collectives is None
                and self.offload_policy is None
                and not self.readiness_report):
            p_ops_t = (physical_ops if physical_ops.__class__ is tuple
                       else tuple(physical_ops))
            prev_top = self._prev_add_topology
            if (prev_top is not None and not external and not prev_top[1]
                    and len(p_ops_t) == len(prev_top[0])):
                same_top = True
                for op, pop in zip(p_ops_t, prev_top[0]):
                    if op is pop:
                        continue
                    if (op.op_id != pop.op_id or op.deps != pop.deps
                            or (op.family == "token_commit")
                            != (pop.family == "token_commit")
                            or not _lazy_safe(op)):
                        same_top = False
                        break
                if same_top:
                    self._prev_add_topology = (p_ops_t, (), prev_top[2])
                    return self._add_lazy(
                        work, replica, release_ms, scope=scope,
                        schedule=schedule, prepared=prepared,
                        input_ready=input_ready, physical_ops=p_ops_t,
                        prefix=prefix, external_inputs=external,
                        emitted_tokens=emitted_tokens,
                        memory_bindings=memory_bindings,
                        state_accesses=state_accesses,
                        verified_dag=prev_top[2])
            known = {op.op_id for op in p_ops_t}
            deps_known = all(
                dep in known or dep in external
                for op in p_ops_t for dep in op.deps)
            if deps_known and all(_lazy_safe(op) for op in p_ops_t):
                res_step = self._add_lazy(
                    work, replica, release_ms, scope=scope,
                    schedule=schedule, prepared=prepared,
                    input_ready=input_ready, physical_ops=p_ops_t,
                    prefix=prefix, external_inputs=external,
                    emitted_tokens=emitted_tokens,
                    memory_bindings=memory_bindings,
                    state_accesses=state_accesses)
                if not external and self._prev_lazy_dag is not None:
                    self._prev_add_topology = (
                        p_ops_t, (), self._prev_lazy_dag[1])
                return res_step
        from .kernel_groups import KernelGroups, declaration
        kernel_groups = KernelGroups(self, physical_ops, replica, prefix)
        for op in physical_ops:
            original = op
            op, recipe = self.prepare(op, replica)
            stage_key = replica, op.stage
            name = prefix + ":" + op.op_id
            names = self.stage_names[stage_key]
            deps = [input_ready, *(ids[key] for key in op.deps)]
            if self.offload_policy is not None and self.offload_policy.expert_ddr_ratio > 0.0:
                from .offload import is_routed_expert_entry_op, schedule_expert_layer_offload
                if is_routed_expert_entry_op(op):
                    exp_key = (op.stage, op.layer)
                    if exp_key not in expert_layer_events:
                        expert_layer_events[exp_key] = schedule_expert_layer_offload(
                            self, prefix, replica, op.stage, op.layer, work.valid_tokens,
                            last_expert_done.get(op.stage, input_ready), op=op)
                    if expert_layer_events[exp_key] is not None:
                        deps.append(expert_layer_events[exp_key])
            if self.offload_policy is None and state_accesses is None:
                bound_accesses = ()
            else:
                bound_accesses = bind_op_accesses(
                    work, op, state_accesses=state_accesses)
                all_bound_accesses.extend(bound_accesses)
            if (self.offload_policy is not None
                    and (self.offload_policy.kvc_ddr_ratio + self.offload_policy.kvc_ssd_ratio) > 0.0):
                from .offload import schedule_kv_layer_offload
                reads: dict[str, list] = {}
                for bound in bound_accesses:
                    access = bound.access
                    if access.direction != "read" or access.bytes <= 0.0:
                        continue
                    reader = access.reader or op.op_id
                    reads.setdefault(reader, []).append(bound)
                for reader, group in reads.items():
                    accesses = tuple(bound.access for bound in group)
                    edges = tuple(dict.fromkeys(
                        edge for bound in group
                        for edge in mandatory_edges(
                            bound.access, ids, bound.memory_dict())))
                    mandatory = (engine.boundary(
                        f"{name}:access_ready:{reader}", after=edges)
                        if edges else None)
                    copy_events = schedule_kv_layer_offload(
                        self, prefix, replica, op.stage, op.layer, work,
                        input_ready, tuple(deps), layer_entries,
                        prev_kv_done=last_kv_done.get(op.stage),
                        state_accesses=accesses,
                        source_ready=mandatory)
                    if copy_events:
                        deps.extend(copy_events)
                        kv_layer_events.setdefault((op.stage, op.layer), []).extend(
                            copy_events)
            deps.extend(kernel_groups.entry_dependencies(op, ids))
            stream_key = replica, op.stage, op.stream or "main"
            stream_tail = kernel_groups.stream_tail(op, stream_key)
            if stream_tail is not None:
                deps.append(stream_tail)
            context = self._context(op, replica, scope, recipe=recipe)
            context = replace(context, group_buffers=kernel_groups.buffer_scope(op))
            owner = f"replica:{replica}"
            captured = scope == "full" or (
                scope == "piecewise" and bool(dict(op.metadata).get("graph_captured"))
            )
            capture_key = op.stage, dict(op.metadata).get("graph_segment", 0) if scope == "piecewise" else 0
            kernel_group = declaration(op)
            submit = (bool(kernel_group.get("entry")) if kernel_group else
                      bool(context.work or getattr(recipe, "algorithm", None)))
            if submit and self.include_submission:
                if captured and capture_key in graph_launch:
                    # Replay reuses the one real launch created for this
                    # capture. Do not create another start handle per caught
                    # operator; the captured ops all depend on this launch.
                    launch, _start = graph_launch[capture_key]
                else:
                    dies = self.stage_dies[stage_key]
                    duration = self.submission.launch_ms(dies, captured=captured)
                    host_deps = (prepared,) + ((self.hosts[stage_key],) if stage_key in self.hosts else ())
                    if (
                        captured
                        and scope == "piecewise"
                        and self.submission.piecewise_sync_ms > 0.0
                    ):
                        if deps:
                            host_deps = tuple(deps) + host_deps
                        if self.submission_counts["graph_replays"] > 0:
                            duration += float(self.submission.piecewise_sync_ms)
                    launch = engine.submit(name + ":submit", {names["host"]: duration},
                                           after=host_deps, owner=owner)
                    self.hosts[stage_key] = launch
                    self.submission_counts[
                        "graph_replays" if captured else "eager_invocations"] += 1
                    start = engine.start_boundary(
                        launch, name + ":host_submit")
                    if captured:
                        graph_launch[capture_key] = (launch, start)
                    # Register exactly once per actual launch creation. Eager
                    # launches at the same timestamp stay distinct because the
                    # tuple carries the real Event identity.
                    host_submits.append((replica, op.stage, start))
                deps.append(launch)
            ready = engine.submit(name + ":ready", after=tuple(deps) or (schedule,), owner=owner)
            if (self.readiness_report
                    and op.family in {"collective", "allgather", "reduce_scatter",
                                      "allreduce", "transport"}
                    and op.link_bytes > ZERO
                    and dict(op.metadata).get("group_axis") not in {"ep", "local_tp"}):
                from .collective_readiness import bind_pipeline_ready
                ranks = self.topology.ranks(replica, op.stage)
                ready_by_rank = {rank: ready for rank in ranks}
                collective, expected_bytes, expected_dtype = _collective_native_basis(
                    op, len(ranks))
                try:
                    readiness_context = bind_pipeline_ready(
                        self.readiness_report, ready_by_rank,
                        expected_collective=collective,
                        expected_bytes=(int(expected_bytes)
                                        if expected_bytes is not None
                                        and expected_bytes.denominator == 1
                                        else None),
                        expected_dtype=expected_dtype,
                        expected_group_size=len(ranks),
                        connection_id=dict(op.metadata).get(
                            "collective_connection_id"),
                        ordinal=dict(op.metadata).get("collective_call_ordinal"),
                        call_id=dict(op.metadata).get("collective_call_id"))
                except ModelingError as exc:
                    self.readiness_unresolved.append({
                        "op_id": op.op_id, "reason": str(exc)})
                else:
                    self.readiness_observations.append({
                        "op_id": op.op_id, "replica": replica,
                        "ready_by_rank": tuple(sorted(ready_by_rank)),
                        "context": readiness_context,
                    })
            if op.layer >= 0 and op.layer not in layer_entries:
                layer_entries[op.layer] = ready
            ready = kernel_groups.phase_ready(op, ready, name)
            recipe_name = op.recipe or type(recipe).__name__
            self.recipes_used[recipe_name] = self.recipes_used.get(recipe_name, 0) + 1
            from .local_tp_groups import collective_requires_cohort

            cross_dp = collective_requires_cohort(op, self.topology)
            if cross_dp and getattr(recipe, "algorithm", None) and self._collectives is None:
                raise ModelingError("joint_collective_requires_complete_cohort")
            grouped = self._collectives is not None and cross_dp
            if grouped:
                complete = engine.future(name + ":collective")
                op_meta = dict(op.metadata)
                grp_ranks = op_meta.get("group_ranks")
                coll_key = (
                    f"{op.op_id}@{','.join(map(str, grp_ranks))}"
                    if grp_ranks and op_meta.get("group_axis") == "local_tp"
                    else op.op_id
                )
                self._collectives.setdefault(coll_key, []).append(CollectiveParticipant(
                    replica, ready, context, recipe, name, complete))
            else:
                physical_hbm = self.stage_pools.get(
                    stage_key, {}).get("hbm", "hbm")
                complete = self._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica,
                    memory_ledger=self.memory,
                    memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(op, memory_bindings))
            complete = kernel_groups.finish(op, context, recipe, complete, stream_key)
            ids[op.op_id] = complete
            if self.offload_policy is not None and self.offload_policy.expert_ddr_ratio > 0.0:
                from .offload import is_routed_expert_entry_op
                if is_routed_expert_entry_op(op) or op.recipe in {"grouped_down", "balanced_grouped_down"}:
                    last_expert_done[op.stage] = complete
            if (self.offload_policy is not None
                    and (self.offload_policy.kvc_ddr_ratio + self.offload_policy.kvc_ssd_ratio) > 0.0):
                from .offload import (
                    schedule_draft_kv_writeback, schedule_kv_layer_writeback,
                )
                target_writes = tuple(
                    bound.access for bound in bound_accesses
                    if bound.access.direction == "write"
                    and bound.access.bytes > 0.0
                    and bound.access.scope != "draft"
                    and bound.access.producer == op.op_id)
                if target_writes:
                    last_kv_done[op.stage] = complete
                    copy_events = schedule_kv_layer_writeback(
                        self, prefix, replica, op.stage, op.layer, work, complete,
                        state_accesses=target_writes)
                    if copy_events:
                        kv_writeback_events.extend(copy_events)
                        owned_offload.extend(copy_events)
                draft_writes = tuple(
                    bound.access for bound in bound_accesses
                    if bound.access.direction == "write"
                    and bound.access.bytes > 0.0
                    and bound.access.scope == "draft"
                    and bound.access.producer == op.op_id)
                if draft_writes:
                    last_draft_done[op.stage] = complete
                    pending_draft.setdefault(op.stage, []).extend(draft_writes)
            if self.on_operator is not None:
                self.on_operator(original, replica, "JointNativeService" if context.joint_native_service
                                 is not None else type(recipe).__name__, ready, complete)
            if op.family == "token_commit":
                declared_commits.update(
                    (request, complete) for request in op.requests)
            for key in self.work:
                self.work[key] += float(getattr(op, key))
        if (self.offload_policy is not None
                and (self.offload_policy.kvc_ddr_ratio
                     + self.offload_policy.kvc_ssd_ratio) > 0.0):
            from .offload import (
                schedule_draft_kv_writeback, schedule_kv_iteration_offload,
            )
            for stage, writes in sorted(pending_draft.items()):
                producer_done = last_draft_done.get(stage)
                if producer_done is None:
                    continue
                copy_events = schedule_draft_kv_writeback(
                    self, prefix, replica, stage, work, producer_done,
                    state_accesses=tuple(writes))
                if copy_events:
                    kv_writeback_events.extend(copy_events)
                    owned_offload.extend(copy_events)
            # Complete physical submit: unissued writes without a named writer
            # still get their own producer Event; a target tail is not the end
            # of the batch. The work registry, not a layer counter, decides
            # what was already issued.
            access_union = tuple(
                bound.access for bound in all_bound_accesses)
            iteration_events = schedule_kv_iteration_offload(
                self, prefix, replica, work, input_ready, physical_ops, ids,
                scheduled_layers=set(kv_layer_events) | kv_writeback_layers,
                state_accesses=access_union,
                scheduled_accesses=None)
            if iteration_events:
                kv_writeback_events.extend(iteration_events)
                owned_offload.extend(iteration_events)
        offload_tails: tuple[Event, ...] = tuple(
            ev for evs in kv_layer_events.values() for ev in evs
        ) + tuple(kv_writeback_events)
        device_complete = engine.submit(
            prefix + ":device_complete", owner=f"replica:{replica}",
            after=(tuple(ids[op.op_id] for op in physical_ops)
                   + offload_tails) or (input_ready,))
        from .offload import release_offload_access
        offload_release = engine.submit(
            prefix + ":offload_release",
            after=(device_complete, *offload_tails),
            owner=f"replica:{replica}", native_hook=True,
            on_complete=lambda _event: release_offload_access(self, work))
        completion = engine.submit(prefix + ":complete", after=(device_complete,),
                                   latency=self.submission.completion_ms if self.include_submission else 0,
                                   owner=f"replica:{replica}")
        fallback_commits = {
            view.request_id: completion for view in work.plan.requests
            if view.request_id not in declared_commits}
        commits = {**fallback_commits, **declared_commits}
        host_results, clients, host_complete = self.serving_io.output(
            self, prefix, replica, work, commits, completion,
            emitted_tokens if emitted_tokens is not None else {
                view.request_id: int(view.draft_advanced_tokens or 1)
                for view in work.plan.requests if view.samples})
        if self.serving_io.enabled:
            completion = engine.submit(prefix + ":step_ready", after=(completion, host_complete),
                                       owner=f"replica:{replica}")
        step = ScheduledIteration(replica, work.plan.iteration, completion, release_ms,
                                  work.mode, work.valid_tokens, work.padded_tokens,
                                  work.graph_bucket, scope, len(physical_ops), commits,
                                  device_complete, host_results, clients,
                                  work.implementation_selections,
                                  tuple(dict.fromkeys(host_submits)),
                                  dict(declared_commits),
                                  dict(fallback_commits),
                                  offload_release)
        engine.submit(prefix + ":step_metrics", after=(completion,),
                      on_complete=lambda _: self.step_summary.record(step),
                      native_hook=True)
        return step

    def prepare_next(self, replica):
        if self.include_submission and self.case.runtime.async_scheduling:
            self.prepared[replica] = self.engine.submit(
                f"{self.namespace}r{replica}:prepare_next",
                {f"{self.namespace}scheduler@r{replica}": self.submission.scheduler_ms},
                after=(self.schedules[replica],), owner=f"replica:{replica}")

    def submission_summary(self) -> dict:
        return {
            "included": self.include_submission,
            **self.submission_counts,
            "host_service_ms": sum(
                self.engine.serviced.get(names["host"], 0)
                for names in self.stage_names.values()),
            "stages": [
                {"replica": replica, "stage": stage,
                 "kernel_launch_ms": self.submission.launch_ms(dies),
                 "graph_replay_ms": self.submission.launch_ms(dies, captured=True),
                 "kernel_parameter": (f"{self.submission_source}.kernel_launch_ms"
                                      if self.submission.kernel_launch_ms is not None
                                      else "machine.device.launch_us"),
                 "replay_parameter": (f"{self.submission_source}.graph_replay_ms"
                                      if self.submission.graph_replay_ms is not None
                                      else "kernel_launch_parameter")}
                for (replica, stage), dies in self.stage_dies.items()
            ],
            "boundary": "host enqueue service; may overlap ready device work",
            "zero_parameter": "excludes undeclared host cost; not measured zero overhead",
        }

    def native_group_summary(self) -> list:
        """Bounded actual launch evidence from the shared native cohorts."""
        groups = []
        for key in sorted(self.native_group_launches):
            item = self.native_group_launches[key]
            groups.append({
                "op_id": item["op_id"], "kind": item["kind"],
                "launch_mode": item.get("launch_mode", ""),
                "ranks": list(item["ranks"]),
                "payload_basis": dict(sorted(item["payload_basis"].items())),
                "payload_packet_bytes": dict(sorted(
                    item["payload_packet_bytes"].items())),
                "payload_frames": {
                    name: dict(sorted(frames.items()))
                    for name, frames in sorted(item["payload_frames"].items())},
                "link_by_phase_by_rank": {
                    name: {phase: {pipe: float(amount)
                                   for pipe, amount in values.items()}
                           for phase, values in sorted(phases.items())}
                    for name, phases in sorted(
                        item["link_by_phase_by_rank"].items())},
                "work_by_rank": {
                    name: [[str(pipe), float(amount)] for pipe, amount in work]
                    for name, work in sorted(item["work_by_rank"].items())},
                "transport_by_rank": {
                    name: {phase: dict(details)
                           for phase, details in sorted(phases.items())}
                    for name, phases in sorted(
                        item.get("transport_by_rank", {}).items())},
                "transport_work_by_rank": {
                    name: [[str(pipe), float(amount)]
                           for pipe, amount in sorted(work.items())]
                    for name, work in sorted(
                        item.get("transport_work_by_rank", {}).items())},
                "transport_tx_bytes": float(item.get(
                    "transport_tx_bytes", 0.0)),
                "transport_rx_bytes": float(item.get(
                    "transport_rx_bytes", 0.0)),
                "transport_frames": int(item.get("transport_frames", 0)),
                "transport_edges": int(item.get("transport_edges", 0)),
                "total_edges": int(item.get("total_edges", 0)),
                "payload_edges": int(item.get("payload_edges", 0)),
                "count_edges": int(item.get("count_edges", 0)),
                "directed_edges": item["directed_edges"],
                "directed_frames": item["directed_frames"],
                "protocol_edges": item["protocol_edges"],
                "unresolved_edges": item["unresolved_edges"],
                "partial": list(item["partial"]),
            })
            if len(groups) >= 64:
                break
        return groups

    def summary(self) -> dict:
        native_groups = self.native_group_summary()
        return {
            **self.engine.summary(), "work": dict(self.work),
            **({"work_basis": "logical_and_analytical_reference; executed_service_is_in_serviced"}
               if self.cost.joint_native else {}),
            "recipes": dict(self.recipes_used),
            "submission": self.submission_summary(),
            "serving_io": self.serving_io.summary(),
            "step_summary": self.step_summary.to_dict(),
            "rank_model": "representative_local_shard_per_stage",
            **({"native_group_count": len(self.native_group_launches),
                "native_groups": native_groups} if native_groups else {}),
            **({"collective_readiness_observations": list(self.readiness_observations),
                "collective_readiness_unresolved": list(self.readiness_unresolved)}
               if (self.readiness_observations or self.readiness_unresolved) else {}),
            **({"joint_collective_observations": list(
                    self._joint_collective_trace.values()),
                "joint_collective_stats": dict(self.joint_collective_stats)}
               if (self._joint_collective_trace is not None
                   and self.joint_collective_stats["observations"]) else {}),

        }
