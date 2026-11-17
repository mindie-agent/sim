"""Declaration-scoped memo for immutable structural evaluation projections.

[Role]
  Serve the repeated evaluation of one already-composed case declaration.  A
  target graph, its Engine execution projection and its isolated operator rows
  are pure functions of the declaration, the probe geometry and the frozen
  execution bindings; none of them contains a clock, a mutable ``CostLedger``,
  an Event bound to a live Engine or any device state.  This module stores only
  those immutable projections so a second evaluation of the same declaration
  does not rebuild the frontend or replay 4,597 operators.

[Invariants]
  * Identity is a content digest.  The selector begins with the SHA-256 of the
    complete parsed declaration, so a changed declaration, hardware rate,
    kernel lowering, workload or SLO always misses.  A mutated raw mapping is
    re-parsed and re-digested, never served from a stale entry.
  * Nothing mutable is cached: no Engine, no Event, no ledger, no allocation
    window, no ``now``.  The stored payload is a plain ``dict``/``tuple`` tree
    that the caller projects into fresh top-level containers.
  * Every call still constructs exactly one ``EvaluationContext`` (and therefore
    exactly one ``CostModel`` and one ``ExecutionBindings``); the memo only
    removes repeated *lowering and replay*, never the per-call frozen bindings.
  * The selector also carries the probe geometry, the replay scope, the external
    input boundary, the execution-boundary flags and the frozen binding
    signature.  A caller that supplies its own recipes, primitives, operator
    report, graph report or observations is never served from the memo.
  * Bounded LRU by entry count.  No I/O, no locks, no global device state.

[Timeouts & Locks]
  Single-threaded bounded ``OrderedDict``; no I/O and no locks.

[Side-Effect Policy]
  Mutates only this module's bounded memo.  Returned payloads are shallow
  projections whose rows the caller must treat as read-only.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Any, Callable


#: Bounded number of declarations retained.  A spine benchmark needs one; a
#: sweep over a handful of geometries needs a few.  Old generations are evicted.
DEFAULT_LIMIT = 8
ZERO = 0
ONE = 1
ASCII = "utf-8"

#: Geometry fields that select a distinct structural projection.
GEOMETRY_FIELDS = (
    "phase", "tokens", "context", "sequences", "scope", "run_graph",
    "include_submission", "replay_scope",
)


def declaration_digest(declaration: Any) -> str:
    """SHA-256 of the complete declaration's canonical text projection."""
    return hashlib.sha256(repr(declaration).encode(ASCII)).hexdigest()


class StructuralMemo:
    """Bounded digest-keyed memo of immutable structural projections."""

    __slots__ = ("_data", "limit", "hits", "misses", "evictions")

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self.limit = max(ONE, int(limit))
        self._data: "OrderedDict[tuple, Any]" = OrderedDict()
        self.hits = ZERO
        self.misses = ZERO
        self.evictions = ZERO

    def get(self, selector: tuple) -> Any:
        try:
            payload = self._data.pop(selector)
        except KeyError:
            self.misses += ONE
            return None
        self._data[selector] = payload
        self.hits += ONE
        return payload

    def put(self, selector: tuple, payload: Any) -> Any:
        if selector in self._data:
            self._data.pop(selector)
        self._data[selector] = payload
        while len(self._data) > self.limit:
            self._data.popitem(last=False)
            self.evictions += ONE
        return payload

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict:
        return {"entries": len(self._data), "limit": self.limit,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


def geometry_key(values: dict, external_inputs: Any = None) -> tuple:
    """Canonical geometry selector for one structural projection."""
    return tuple(values.get(field) for field in GEOMETRY_FIELDS) + (
        _freeze(external_inputs),)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item))
                            for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def project(payload: Any) -> Any:
    """Fresh top-level containers over an immutable memoized payload."""
    if isinstance(payload, tuple):
        return tuple(project(item) for item in payload)
    if isinstance(payload, dict):
        return {key: (list(value) if isinstance(value, list) else value)
                for key, value in payload.items()}
    return payload


def lookup(memo: StructuralMemo, selector: tuple) -> Any:
    return memo.get(selector)


def store(memo: StructuralMemo, selector: tuple, build: Callable[[], Any]) -> Any:
    return memo.put(selector, build())
