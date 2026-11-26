"""The one mutable execution root for a causal runtime.

[Role] Own every mutable array of one causal universe: live tasks, frontier
  heaps, event counters, resource occupancy, capacity domains, state versions,
  host cursors, program cursors and bounded observers.
[Invariants] Exactly one owner runtime per domain; Engine is the only writer of
  the clock/frontier/transaction fields. Memory, conductors and executions hold
  views into this object, never a second allocation/state dictionary.
[Timeouts & Locks] None; all mutation happens on the calling Engine thread.
[Side-Effect Policy] Construction allocates the bounded containers; helpers
  here expose ownership/claim/snapshot only.
"""
from __future__ import annotations

from collections import deque

from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from .contracts import CapacityDomainState


#: Immutable empty pool set shared by runtimes that declare no memory domain.
_EMPTY_FROZENSET: frozenset = frozenset()


#: Allocation evidence is an independent bounded audit stream.  It is not the
#: opt-in task trace: disabling the verbose ``trace`` must never change what
#: allocation evidence a settled run reports.
ALLOCATION_TRACE_CAPACITY = 8192
DEFAULT_ZERO_PROGRESS_LIMIT = 100_000


class _LazyTraceDeque:
    """Bounded ring buffer materializing task trace dicts only when read."""

    __slots__ = ("_dq", "maxlen", "_ext_mut")

    def __init__(self, iterable=(), maxlen: int | None = None):
        self.maxlen = maxlen
        self._dq = deque(iterable, maxlen=maxlen)
        self._ext_mut = 0

    @staticmethod
    def _to_dict(row):
        if row.__class__ is dict:
            return row
        name, owner, start, end, demand, service, meta, placement, coupled = row
        return {
            "task": name,
            "owner": owner,
            "start_ms": start,
            "end_ms": end,
            "demand": demand.copy() if demand else {},
            "serviced": service,
            "metadata": meta if meta.__class__ is dict else (dict(meta) if meta else {}),
            "placement": placement.copy() if placement else {},
            "coupled": coupled,
        }

    def append(self, item) -> None:
        self._dq.append(item)
        self._ext_mut += 1

    def clear(self) -> None:
        self._dq.clear()
        self._ext_mut += 1

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return bool(self._dq)

    def __iter__(self):
        return (self._to_dict(r) for r in self._dq)

    def __reversed__(self):
        return (self._to_dict(r) for r in reversed(self._dq))

    def __getitem__(self, index):
        return self._to_dict(self._dq[index])

    def __eq__(self, other):
        if isinstance(other, (_LazyTraceDeque, deque, list)):
            return list(self) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(deque((self._to_dict(r) for r in self._dq), maxlen=self.maxlen))


class _LazyAllocTraceDeque:
    """Bounded ring buffer materializing allocation audit dicts only when read."""

    __slots__ = ("_dq", "maxlen", "_ext_mut")

    def __init__(self, iterable=(), maxlen: int | None = None):
        self.maxlen = maxlen
        self._dq = deque(iterable, maxlen=maxlen)
        self._ext_mut = 0

    @staticmethod
    def _to_dict(row):
        if row.__class__ is dict:
            return row
        t_ms, is_acq, item, tname = row
        return {
            "time": t_ms,
            "action": "acquire" if is_acq else "release",
            "name": item.name,
            "pool": item.pool,
            "size": item.size,
            "units": item.units,
            "task": tname,
        }

    def append(self, item) -> None:
        self._dq.append(item)
        self._ext_mut += 1

    def clear(self) -> None:
        self._dq.clear()
        self._ext_mut += 1

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return bool(self._dq)

    def __iter__(self):
        return (self._to_dict(r) for r in self._dq)

    def __reversed__(self):
        return (self._to_dict(r) for r in reversed(self._dq))

    def __getitem__(self, index):
        return self._to_dict(self._dq[index])

    def __eq__(self, other):
        if isinstance(other, (_LazyAllocTraceDeque, deque, list)):
            return list(self) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(deque((self._to_dict(r) for r in self._dq), maxlen=self.maxlen))


class RuntimeState:
    def __init__(
        self, *, resources: Mapping[str, Any], pools=None, trace_limit: int = 0,
        pool_units=None, zero_progress_limit: int = DEFAULT_ZERO_PROGRESS_LIMIT,
    ):
        self.uid = uuid4().hex
        self.capacity_domains: dict[str, CapacityDomainState] = {}
        self.state_versions: dict[str, list[Any]] = {}
        self.cache_domains: dict[str, Any] = {}
        self.program_cursors: dict[int, Any] = {}
        self.occurrence_metrics: dict[str, dict[str, Any]] = {}
        self.host_fifos: dict[str, Any] = {}
        self.dirty_resource_components: set[str] = set()
        self.bounded_catalog_cache: dict[Any, Any] = {}
        self.periodic_baselines: dict[str, Any] = {}
        self.online_observers: dict[str, Any] = {}
        self.resources = dict(resources)
        self.pools = dict(pools or {})
        self.pool_units = {name: (pool_units or {}).get(name, 1) for name in self.pools}
        self.zero_progress_limit = zero_progress_limit
        # Optional T08 Memory budget. None keeps pure hardware-only contexts.
        self.memory = None
        self.memory_pools: set[str] = set()
        #: Frozen view of ``memory_pools``; rebuilt only when the set changes.
        self._memory_pools_frozen: frozenset[str] = _EMPTY_FROZENSET
        #: Interned Engine-owned pool sets created by ``memory_scope``.
        self._engine_pool_sets: dict[tuple[str, ...], frozenset[str]] = {}
        self._engine_pool_set_ids: set[int] = set()
        # Actual granted allocation name -> binding/replica/ledger. The grant
        # stays live until the allocation's real free boundary, not when the
        # acquiring task happens to complete.
        self._memory_grants: dict[str, dict] = {}
        self._memory_pending: set[str] = set()
        self._memory_ledgers: dict[int, object] = {}
        self.allocated = dict.fromkeys(self.pools, 0.0)
        self.peak_allocated = dict(self.allocated)
        self._unit_allocated: dict = {}
        self._unit_alloc_dirty: bool = False
        self._pool_unit_masks: dict[str, int] = dict.fromkeys(self.pools, 0)
        self._pair_rate_cache: dict = {}
        self.peak_unit_allocated = {}
        self.allocations: dict[str, Allocation] = {}
        self.now = 0.0
        self.serviced = dict.fromkeys(self.resources, 0.0)
        self.delivered: dict[str, float] = {}
        self.busy_ms = dict.fromkeys(self.resources, 0.0)
        self._tasks: dict[int, _Task] = {}
        self._promises: dict[int, Event] = {}
        self._start_waiters: dict[int, list[Event]] = {}
        # Ready and completable heaps are sequence-ordered. Running tasks stay
        # out of the completable heap until work and latency are both done, so
        # a zero-work chain does not re-sort every overlapping active task.
        # Admission-ready entries are (priority, sequence): lower priority
        # values are admitted first, then submission order. The Engine never
        # preempts a started task.
        self._ready_heap: list[tuple[int, int]] = []
        self._ready_seen: set[int] = set()
        self._completable: list[int] = []
        self._completable_seen: set[int] = set()
        self._active: set[int] = set()
        #: Active tasks that still owe a declared latency interval.  A task
        #: with ``latency == 0`` is pushed to the completable heap the moment
        #: its remaining work empties, so no step needs to scan every active
        #: task to discover a finished one.
        self._latency_active: set[int] = set()
        #: Active-task counters for the pure arbiter fast paths: an Engine with
        #: no coupled task skips the coupled solver, and one with no placement
        #: skips per-resource pinned partitions.
        self._coupled_active = 0
        self._placed_active = 0
        self._delivery_live = 0
        #: Structural cache of validated immutable Stage declarations.  Only
        #: static work/caps/placement validation is shared; every occurrence
        #: still binds and validates its own scoped allocations.
        self._stage_prepared: dict[tuple, Any] = {}
        self._stage_prepared_id: dict[int, tuple] = {}
        self._releases: list[tuple[float, int]] = []
        self._max_release: float = 0.0
        self._sequence = 0
        self._next_launch = 0
        # Periodic acceleration registries are populated by the template
        # producer (scheduling.start_pipeline_template). The detector is a pure
        # read-only helper; this Engine remains the only clock/state owner.
        self._periodic_runs: dict[int, dict] = {}
        self._periodic_tasks: dict[int, tuple] = {}
        self._periodic_detector = None
        self._periodic_last = None
        self._periodic_stall_epoch: tuple | None = None
        self._periodic_stall_obs = 0
        self._periodic_last_next_sum = -1
        self._periodic_no_issue_streak = 0
        #: Monotone mutation counter for the allocation/grant frontier.  Every
        #: acquire/release of a live buffer or memory grant bumps it, so the
        #: periodic observer can reuse its previous structural view instead of
        #: re-walking persistent weights and KV grants on every micro-step.
        self._alloc_state_version = 0
        self._periodic_alloc_cache: tuple | None = None
        #: Non-periodic tasks that really completed.  A plain counter is
        #: sufficient evidence of external interference inside one interval
        #: and never scans the whole blocked frontier.
        self._external_completions = 0
        self.completed_count = 0
        self.peak_tasks = 0
        self._trace_view = _LazyTraceDeque(maxlen=trace_limit)
        self._trace_dq = self._trace_view._dq
        self._trace_maxlen = 0 if trace_limit is None else int(trace_limit)
        self._trace_appends = 0
        self.trace_dropped = 0
        self._evicted_service_spans: dict[str, list[tuple[float, float]]] = {}
        # One authoritative typed state ledger. FragmentConductor and
        # Execution hold views of this object, never a second version map.
        self.state_ledger = None
        # Bounded allocation event evidence; default aggregation keeps counts
        # only. Completed objects themselves are still reclaimed immediately.
        self._alloc_trace_view = _LazyAllocTraceDeque(maxlen=ALLOCATION_TRACE_CAPACITY)
        self._alloc_trace_dq = self._alloc_trace_view._dq
        self._alloc_trace_maxlen = ALLOCATION_TRACE_CAPACITY
        self.allocation_event_count = 0
        self.advance_calls = 0
        self.advance_steps = 0
        self._settle_rounds = 0
        self._compact_step = False

    def _sync_unit_allocated(self) -> dict:
        if self._unit_alloc_dirty:
            ua = self._unit_allocated
            ua.clear()
            for rem_a in self.allocations.values():
                rpool = rem_a.pool
                rsize = rem_a.size
                for u in rem_a.units:
                    pair = (rpool, u)
                    ua[pair] = ua.get(pair, 0.0) + rsize
            self._unit_alloc_dirty = False
        return self._unit_allocated

    @property
    def unit_allocated(self) -> dict:
        return self._sync_unit_allocated()

    @unit_allocated.setter
    def unit_allocated(self, value: dict) -> None:
        self._unit_allocated = value
        self._unit_alloc_dirty = False
        pm = dict.fromkeys(self.pools, 0)
        for (p, u), amt in value.items():
            if amt:
                pm[p] = pm.get(p, 0) | (1 << u)
        self._pool_unit_masks = pm

    @property
    def trace(self):
        return self._trace_view

    @trace.setter
    def trace(self, value) -> None:
        if isinstance(value, _LazyTraceDeque):
            self._trace_view = value
        elif isinstance(value, deque):
            self._trace_view = _LazyTraceDeque(value, maxlen=value.maxlen)
        else:
            self._trace_view = _LazyTraceDeque(
                value, maxlen=getattr(value, "maxlen", None))
        self._trace_dq = self._trace_view._dq
        ml = self._trace_view.maxlen
        self._trace_maxlen = -1 if ml is None else int(ml)
        self._trace_appends = 0

    @property
    def allocation_trace(self):
        return self._alloc_trace_view

    @allocation_trace.setter
    def allocation_trace(self, value) -> None:
        if isinstance(value, _LazyAllocTraceDeque):
            self._alloc_trace_view = value
        elif isinstance(value, deque):
            self._alloc_trace_view = _LazyAllocTraceDeque(value, maxlen=value.maxlen)
        else:
            self._alloc_trace_view = _LazyAllocTraceDeque(
                value, maxlen=getattr(value, "maxlen", None))
        self._alloc_trace_dq = self._alloc_trace_view._dq
        ml = self._alloc_trace_view.maxlen
        self._alloc_trace_maxlen = -1 if ml is None else int(ml)

    # -- Public causal-root identity -------------------------------------
    @property
    def now_ms(self) -> float:
        return self.now

    @property
    def next_sequence(self) -> int:
        return self._sequence

    @property
    def next_launch(self) -> int:
        return self._next_launch

    def ensure_state_ledger(self, initial: Mapping[str, int] | None = None, *,
                            version_limit: int = 64):
        """Return the one typed state ledger owned by this runtime root.

        Consumers (``Execution.fragment_state`` and every
        ``FragmentConductor``) receive this exact object, so a catalog commit
        and a fragment commit cannot disagree. ``initial`` only seeds targets
        that are not already known; a conflicting version is an error rather
        than a silent overwrite. A ledger already owned by another runtime is
        rejected instead of being shared as a second truth.
        """
        from ..scheduling import StateLedger

        candidate = self.state_ledger
        if candidate is None:
            candidate = (initial if isinstance(initial, StateLedger)
                         else StateLedger(initial, version_limit=version_limit))
        elif initial is not None:
            if isinstance(initial, StateLedger):
                if initial is not candidate:
                    raise ValueError("state_ledger_conflict:explicit")
            else:
                candidate.initialize(initial)
        owner = getattr(candidate, "runtime_uid", None)
        if owner is None:
            candidate.runtime_uid = self.uid
        elif owner != self.uid:
            # Reject before installing: a failed adoption must not leave a
            # foreign ledger attached to this runtime.
            raise ValueError(f"state_ledger_runtime_conflict:{owner}")
        self.state_ledger = candidate
        return candidate

    def claim_capacity_domain(self, domain: CapacityDomainState) -> None:
        owner = domain.owner_runtime_uid
        if owner is not None and owner != self.uid:
            raise ValueError("memory_runtime_conflict")
        existing = self.capacity_domains.get(domain.domain_id)
        if existing is not None and existing is not domain:
            raise ValueError(f"capacity_domain_conflict:{domain.domain_id}")
        domain.owner_runtime_uid = self.uid
        self.capacity_domains[domain.domain_id] = domain

    def snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType({
            "uid": self.uid,
            "now_ms": self.now,
            "next_sequence": self._sequence,
            "live_tasks": len(self._tasks),
            "completed_tasks": self.completed_count,
            "serviced": MappingProxyType(dict(self.serviced)),
            "busy_ms": MappingProxyType(dict(self.busy_ms)),
            "capacities": MappingProxyType(dict(self.pools)),
            "allocated": MappingProxyType(dict(self.allocated)),
            "peak_allocated": MappingProxyType(dict(self.peak_allocated)),
            "state_versions": len(self.state_versions),
            "program_cursors": len(self.program_cursors),
        })
