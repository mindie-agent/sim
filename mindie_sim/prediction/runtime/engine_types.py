"""Core value types, task structures and binding helpers for the causal Engine.

[Role] Define immutable resource/allocation descriptors, Event handles, prepared
  task templates, live task records and memory-binding helpers shared across the
  runtime Engine mixins and the native C11 ABI bridge.
[Invariants] ``Resource`` validates finite positive rates and unit counts at
  construction; ``_PreparedTask`` is frozen and reusable across repeated tiles;
  empty sentinel maps are read-only.
[Timeouts & Locks] Pure in-memory data structures; no locks, sleeps or I/O.
[Side-Effect Policy] Mutates only a local bounded unit-mask memo.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping

class ExecutionError(ValueError):
    pass


#: Small repeated catalogs are fully validated before publication. Larger ones
#: validate their declared live window up front and each remaining panel at the
#: real activation boundary, so preflight stays independent of ``finite.total``.
_FULL_FINITE_PREFLIGHT = 4096

_EMPTY_FROZENSET: frozenset = frozenset()
#: Shared immutable-empty declarations for prepared tasks that declare neither
#: a supply cap nor a placement.  ``_Task`` only ever reads them.
_EMPTY_CAPS: dict = {}
_EMPTY_PLACEMENT: dict = {}
_EMPTY_METADATA: dict = {}


@dataclass(frozen=True)
class Resource:
    name: str
    rate: float
    unit: str = "work"
    policy: str = "fair"
    units: int = 1

    def __post_init__(self):
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ExecutionError(f"resource_rate_invalid:{self.name}")
        if self.policy not in {"fair", "fifo"}:
            raise ExecutionError(f"resource_policy_invalid:{self.policy}")
        if self.units < 1:
            raise ExecutionError(f"resource_units_invalid:{self.name}")


@dataclass(unsafe_hash=True, slots=True)
class Allocation:
    name: str
    pool: str
    size: float
    units: tuple[int, ...] = (0,)


_UNITS_MASK_CACHE: dict[tuple[int, ...], int] = {}


def _units_to_mask(units: tuple[int, ...]) -> int:
    mask = _UNITS_MASK_CACHE.get(units)
    if mask is not None:
        return mask
    m = 0
    for u in units:
        m |= 1 << u
    if len(_UNITS_MASK_CACHE) >= 4096:
        _UNITS_MASK_CACHE.clear()
    _UNITS_MASK_CACHE[units] = m
    return m


def _compute_pool_unit_masks(
    acquire: tuple[Allocation, ...],
) -> tuple[tuple[str, int], ...] | None:
    if not acquire:
        return ()
    if len(acquire) == 1:
        a0 = acquire[0]
        if a0.size <= 0:
            return None
        return ((a0.pool, _units_to_mask(a0.units)),)
    masks: dict[str, int] = {}
    for a in acquire:
        if a.size <= 0:
            return None
        m = _units_to_mask(a.units)
        p = a.pool
        masks[p] = masks.get(p, 0) | m
    return tuple(masks.items())


class Event:
    __slots__ = ("sequence", "name", "time", "started", "_waiters", "_callbacks")

    def __init__(self, sequence: int, name: str,
                 time: float | None = None, started: float | None = None):
        self.sequence = sequence
        self.name = name
        self.time = time
        self.started = started
        self._waiters: list[int] = []
        self._callbacks: list | None = None

    @property
    def done(self) -> bool:
        return self.time is not None

    def __repr__(self) -> str:
        return (f"Event(sequence={self.sequence!r}, name={self.name!r}, "
                f"time={self.time!r}, started={self.started!r})")


@dataclass(slots=True)
class _Task:
    event: Event
    remaining: dict[str, float]
    demand: dict[str, float]
    waiting: int
    release: float
    latency: float
    caps: dict[str, float]
    acquire: tuple[Allocation, ...]
    free: tuple[Allocation, ...]
    owner: str
    metadata: dict
    placement: dict
    allocation_demands: tuple[tuple[tuple[str, int], float], ...]
    priority: int = 0
    coupled: bool = False
    callback: Callable[[Event], None] | None = None
    start_callback: Callable[[Event], None] | None = None
    start: float | None = None
    service: dict[str, float] = field(default_factory=dict)
    memory_granted: bool = False
    #: Per-task T08 binding captured at submit. A conductor may share one
    #: Engine between PD endpoints with different ledgers, so the ledger is
    #: attached to the task instead of overwriting ``Engine.memory``.
    memory_ledger: object | None = None
    memory_replica: int = 0
    memory_stage: int = 0
    memory_pools: frozenset[str] = frozenset()
    memory_bindings: Mapping[str, Mapping] | None = None
    memory_grant_names: tuple[str, ...] = ()
    cohort_size: int = 1
    prepared: _PreparedTask | None = None
    start_observer: Event | None = None
    #: The completion hook is a pure, bounded frontier expansion: the kernel may
    #: mirror this row and stops at its completion instant (``F_CALLBACK``)
    #: before Python admits the pre-prepared successors.  Only rows whose hook
    #: is virtual-time independent set this.
    native_hook: bool = False
    cohort_widened: tuple[str, ...] = ()
    cohort_members: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedTask:
    """Validated static service; each submission owns its remaining work."""

    demand: dict[str, float]
    latency: float
    caps: dict[str, float]
    acquire: tuple[Allocation, ...]
    free: tuple[Allocation, ...]
    placement: dict[str, tuple[int, ...]]
    coupled: bool
    allocation_demands: tuple[tuple[tuple[str, int], float], ...]
    priority: int = 0
    cohort_size: int = 1
    acquire_assigns: tuple[tuple[tuple[tuple[str, int], float], ...], ...] = ()
    free_pairs: tuple[tuple[tuple[str, int], ...], ...] = ()
    acquire_pools: frozenset[str] = _EMPTY_FROZENSET
    static_id: int = 0
    fits_empty_pools: bool = True
    coupled_demand: dict[str, float] | None = None
    coupled_cap_limit: float = math.inf
    coupled_solo_rates: tuple[tuple[str, float], ...] | None = None
    demand_keys: frozenset = _EMPTY_FROZENSET
    solo_rates: dict[str, float] | None = None
    coupled_res_info: dict[str, tuple[float, float, float, float, tuple[int, ...] | None]] | None = None
    uncoupled_res_info: dict[str, tuple[float, float, float, float, float, tuple[int, ...] | None]] | None = None
    pool_unit_masks: tuple[tuple[str, int], ...] | None = ()
    cohort_widened: tuple[str, ...] = ()
    cohort_members: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = ()


_EMPTY_TASK = _PreparedTask({}, 0.0, {}, (), (), {}, False, (), 0)

_MEMORY_FIELDS = (
    "owner", "storage", "tier", "rank", "stage", "generation", "ranges",
    "event", "host_bytes", "physical_key", "unit", "last_use",
)


def _memory_binding(metadata: Mapping | None) -> dict:
    """Read one allocation's physical binding from task metadata.

    FragmentConductor historically wrote ``memory_owner`` etc. while Engine
    read the bare names. Both spellings now resolve to the same fields; bare
    names win only when both are present and differ is impossible because the
    producer writes one spelling per call.
    """
    data = dict(metadata or {})
    result = {}
    for field in _MEMORY_FIELDS:
        if data.get(field) is not None:
            result[field] = data[field]
        elif data.get("memory_" + field) is not None:
            result[field] = data["memory_" + field]
    return result


def _structural_value(value: Any) -> Any:
    """Hashable structural view for Memory/Persistent frontier state."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Event):
        return ("event", value.sequence, value.name)
    if isinstance(value, (tuple, list)):
        return tuple(_structural_value(item) for item in value)
    if isinstance(value, _AbcMapping):
        return tuple(sorted((str(key), _structural_value(item))
                            for key, item in value.items()))
    return str(value)


def _chain_callbacks(first: Callable[[Event], None],
                     second: Callable[[Event], None]):
    """Run two real completion observers in their original registration order."""
    def chained(event):
        first(event)
        second(event)
    return chained


def _binding_for(name: str, bindings: Mapping[str, Mapping]) -> dict:
    """Resolve an allocation binding by exact name or scoped suffix.

    ``KernelProgram.start`` scopes allocation names as ``name:stage:buffer``
    without exposing that scoped identity to the caller. A unique suffix is
    therefore the producer's declared buffer identity, not a guessed layer or
    event name.
    """
    if not bindings:
        return {}
    exact = bindings.get(name)
    if exact is not None:
        return dict(exact)
    matches = []
    for key, value in bindings.items():
        stem = str(key)
        if (name.endswith(":" + stem) or name.endswith("#" + stem)
                or name.endswith("/" + stem)):
            matches.append((len(stem), dict(value)))
    if not matches:
        return {}
    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ExecutionError(f"memory_binding_ambiguous:{name}")
    return matches[0][1]


PreparedStage = _PreparedTask
