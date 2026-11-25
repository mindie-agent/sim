"""Canonical periodic state and bounded template interning.

[Role] Reuse validated invariant templates across settled frontiers.
[Invariants] A fingerprint only rejects candidates quickly. A cache hit requires
  exact template identity or equality. Retention has a byte limit.
[Timeouts & Locks] Caches are bounded and process-local. No locks or I/O.
[Side-Effect Policy] Mutates template caches and diagnostic counters. Snapshots are frozen.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

from .engine_state import _engine_static_key, _freeze_engine_dynamic
from .snapshots import (
    PeriodicContractError,
    _TEMPLATE_REPR_LIMIT,
    _check_canonical_snapshot,
    _check_structural,
    _freeze_snapshot,
    _is_int,
    _measure_retained,
    _numeric_close,
    _require_int,
    _structural_equal,
)


# Bounded process-local structural template cache.  Old generations are evicted;
# the cache never grows with observation history.  Exact template equality is
# required even on a fingerprint hit.
_DEFAULT_INTERNER_LIMIT = 64
# Conservative local shares of the host cache envelope.  A host (T04) may pass
# the actual remaining byte budget explicitly; the helper never assumes that a
# whole 24/48 MiB host scope is available to one cache.
_DEFAULT_INTERNER_BYTES = 8 * 1024 * 1024
_TEMPLATE_OVERHEAD = 256
_STATE_METRICS = {
    "canonical_frames": 0,
    "raw_frames": 0,
    "template_validations": 0,
    "template_nodes": 0,
    "template_builds": 0,
    "template_build_nodes": 0,
    "template_hits": 0,
    "template_hit_nodes": 0,
    "template_probes": 0,
    "dynamic_validations": 0,
    "dynamic_nodes": 0,
    "raw_nodes": 0,
}
_STATE_VERSION = 0


def reset_metrics() -> None:
    """Reset process-local structural validation counters (diagnostics only)."""
    for key in _STATE_METRICS:
        _STATE_METRICS[key] = 0


def metrics() -> dict:
    """Return a copy of the structural validation counters."""
    return dict(_STATE_METRICS)


def _next_state_version() -> int:
    global _STATE_VERSION
    _STATE_VERSION += 1
    return _STATE_VERSION


def _template_cost(template: tuple, template_repr: str) -> int:
    """Admission cost for one cached template (tree + index/reference bookkeeping)."""
    return _measure_retained(template) + len(template_repr) + _TEMPLATE_OVERHEAD


@dataclass(frozen=True)
class TemplateRef:
    """Immutable validated invariant snapshot with a generation-scoped id.

    ``template`` is a deep-frozen typed canonical tree (lists become tagged
    immutable tuples, dicts/sets/mapping types are tagged and ordered
    deterministically).  Identity and matching use this tree, never display
    text.  ``template_repr`` is a bounded diagnostic preview retained for
    compatibility; it is deliberately not a comparison key.
    A direct constructor validates/freezes a caller builtin tree;
    ``from_validated`` accepts only an already canonical internal tree.
    """

    template_id: int
    generation: Hashable
    fingerprint: int
    template: tuple
    template_repr: str = ""
    nodes: int = 0
    retained_bytes: int = 0

    def __post_init__(self) -> None:
        _require_int("template_id", self.template_id, minimum=1)
        if not isinstance(self.generation, (int, str)) or isinstance(
                self.generation, bool):
            raise PeriodicContractError(
                f"periodic_template_generation_invalid:{self.generation!r}")
        if not _is_int(self.fingerprint):
            raise PeriodicContractError(
                f"periodic_template_fingerprint_invalid:{self.fingerprint!r}")
        if not isinstance(self.template, tuple):
            raise PeriodicContractError("periodic_template_must_be_tuple")
        if not isinstance(self.template_repr, str):
            raise PeriodicContractError("periodic_template_repr_invalid")
        counter = [0]
        _check_structural(self.template, path="template", counter=counter)
        representation = (
            self.template_repr[:_TEMPLATE_REPR_LIMIT]
            if self.template_repr else "")
        snapshot = _freeze_snapshot(self.template, path="template")
        object.__setattr__(self, "template", snapshot)
        object.__setattr__(self, "template_repr", representation)
        object.__setattr__(self, "nodes", counter[0])
        object.__setattr__(
            self, "retained_bytes", _template_cost(snapshot, representation))
        _STATE_METRICS["template_validations"] += 1
        _STATE_METRICS["template_nodes"] += counter[0]

    @classmethod
    def from_validated(
        cls,
        template_id: int,
        generation: Hashable,
        fingerprint: int,
        template: tuple,
        template_repr: str,
        nodes: int,
        retained_bytes: int | None = None,
    ) -> "TemplateRef":
        """Build a ref from an internal, already canonical snapshot only."""
        _require_int("template_id", template_id, minimum=1)
        if not isinstance(generation, (int, str)) or isinstance(generation, bool):
            raise PeriodicContractError(
                f"periodic_template_generation_invalid:{generation!r}")
        if not _is_int(fingerprint):
            raise PeriodicContractError(
                f"periodic_template_fingerprint_invalid:{fingerprint!r}")
        if not isinstance(template, tuple):
            raise PeriodicContractError("periodic_template_must_be_tuple")
        if not isinstance(template_repr, str):
            raise PeriodicContractError("periodic_template_repr_invalid")
        _require_int("template_nodes", nodes, minimum=0)
        _check_canonical_snapshot(template, path="template")
        template_repr = template_repr[:_TEMPLATE_REPR_LIMIT]
        # Cost is always derived from the actual retained tree.  A caller may
        # omit it, but an explicitly supplied value must match exactly: the
        # history cache only accepts internally computed costs.
        computed_bytes = _template_cost(template, template_repr)
        if retained_bytes is not None:
            _require_int("template_retained_bytes", retained_bytes, minimum=0)
            if retained_bytes != computed_bytes:
                raise PeriodicContractError(
                    "periodic_template_retained_bytes_mismatch")
        retained_bytes = computed_bytes
        obj = object.__new__(cls)
        object.__setattr__(obj, "template_id", template_id)
        object.__setattr__(obj, "generation", generation)
        object.__setattr__(obj, "fingerprint", fingerprint)
        object.__setattr__(obj, "template", template)
        object.__setattr__(obj, "template_repr", template_repr)
        object.__setattr__(obj, "nodes", nodes)
        object.__setattr__(obj, "retained_bytes", retained_bytes)
        _STATE_METRICS["template_validations"] += 1
        _STATE_METRICS["template_nodes"] += nodes
        return obj

    @classmethod
    def _from_internal_frozen(
        cls,
        template_id: int,
        generation: Hashable,
        fingerprint: int,
        snapshot: tuple,
        nodes: int,
    ) -> "TemplateRef":
        """Build a ref directly from an internal ``_freeze_snapshot`` result."""
        retained_bytes = _template_cost(snapshot, "")
        obj = object.__new__(cls)
        object.__setattr__(obj, "template_id", template_id)
        object.__setattr__(obj, "generation", generation)
        object.__setattr__(obj, "fingerprint", fingerprint)
        object.__setattr__(obj, "template", snapshot)
        object.__setattr__(obj, "template_repr", "")
        object.__setattr__(obj, "nodes", nodes)
        object.__setattr__(obj, "retained_bytes", retained_bytes)
        _STATE_METRICS["template_validations"] += 1
        _STATE_METRICS["template_nodes"] += nodes
        return obj


class StateInterner:
    """Bounded exact-structure interner for immutable invariant fragments.

    ``intern`` is the only path that validates a new template.  A hit is an
    exact full-canonical-tree match in the fingerprint bucket; a display repr
    or a fingerprint alone is never accepted.  Entries are LRU-evicted at
    ``limit`` and by their measured retained bytes, and all entries from an
    older generation are dropped when the generation changes.  An item larger
    than the remaining byte budget is returned as a live ref but not cached.
    The cache therefore cannot grow with observation history or bypass the
    caller's byte envelope.
    """

    def __init__(
        self,
        *,
        limit: int = _DEFAULT_INTERNER_LIMIT,
        retained_bytes_limit: int = _DEFAULT_INTERNER_BYTES,
    ) -> None:
        self.limit = _require_int("interner_limit", limit, minimum=1)
        self.retained_bytes_limit = _require_int(
            "interner_retained_bytes_limit", retained_bytes_limit, minimum=1)
        self._buckets: dict[int, list[TemplateRef]] = {}
        self._engine_buckets: dict[int, list[tuple[tuple, TemplateRef]]] = {}
        self._refs: dict[int, TemplateRef] = {}
        self._order: OrderedDict[int, None] = OrderedDict()
        self._next_id = 1
        self._generation: Hashable | None = None
        self._retained_bytes = 0
        self._live: weakref.WeakSet = weakref.WeakSet()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.uncached = 0
        self.oversized = 0
        self.rejected = 0

    @property
    def templates(self) -> int:
        return len(self._refs)

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    def _touch(self, ref: TemplateRef) -> None:
        self._order[ref.template_id] = None
        self._order.move_to_end(ref.template_id)

    def _drop(self, ref: TemplateRef) -> None:
        if self._refs.pop(ref.template_id, None) is None:
            return
        self._order.pop(ref.template_id, None)
        self._retained_bytes = max(
            0, self._retained_bytes - ref.retained_bytes)
        bucket = self._buckets.get(ref.fingerprint)
        if bucket is not None:
            self._buckets[ref.fingerprint] = [item for item in bucket if item.template_id != ref.template_id]
            if not self._buckets[ref.fingerprint]:
                del self._buckets[ref.fingerprint]
        engine_bucket = self._engine_buckets.get(ref.fingerprint)
        if engine_bucket is not None:
            self._engine_buckets[ref.fingerprint] = [
                entry for entry in engine_bucket
                if entry[1].template_id != ref.template_id]
            if not self._engine_buckets[ref.fingerprint]:
                del self._engine_buckets[ref.fingerprint]
        self.evictions += 1

    def _evict_oldest(self) -> None:
        while self._order and (
            len(self._refs) > self.limit
            or self._retained_bytes > self.retained_bytes_limit
        ):
            template_id, _ = self._order.popitem(last=False)
            ref = self._refs.get(template_id)
            if ref is not None:
                self._drop(ref)

    def evict_generation(self, generation: Hashable) -> None:
        for ref in tuple(self._refs.values()):
            if ref.generation != generation:
                self._drop(ref)

    def clear(self) -> None:
        self._buckets.clear()
        self._engine_buckets.clear()
        self._refs.clear()
        self._order.clear()
        self._next_id = 1
        self._generation = None
        self._retained_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.uncached = self.oversized = self.rejected = 0

    def intern(
        self,
        template: tuple,
        *,
        generation: Hashable = 0,
        fingerprint: int | None = None,
    ) -> TemplateRef:
        if not isinstance(template, tuple):
            raise PeriodicContractError("periodic_intern_template_must_be_tuple")
        if not isinstance(generation, (int, str)) or isinstance(generation, bool):
            raise PeriodicContractError(f"periodic_intern_generation_invalid:{generation!r}")
        if self._generation != generation:
            self.evict_generation(generation)
            self._generation = generation
        # Type-exact canonicalization happens before any hit is possible.  The
        # display repr and a caller-supplied fingerprint are bucket selectors
        # only; a hit always checks the full canonical tree.
        counter = [0]
        snapshot = _freeze_snapshot(template, path="template", counter=counter)
        if fingerprint is None:
            fingerprint = hash(snapshot)
        if not _is_int(fingerprint):
            raise PeriodicContractError(
                f"periodic_intern_fingerprint_invalid:{fingerprint!r}")
        probes = 0
        for ref in self._buckets.get(fingerprint, ()):
            probes += 1
            if ref.generation == generation and ref.template == snapshot:
                _STATE_METRICS["template_hits"] += 1
                _STATE_METRICS["template_hit_nodes"] += counter[0]
                _STATE_METRICS["template_probes"] += probes
                self.hits += 1
                self._touch(ref)
                return ref
        _STATE_METRICS["template_probes"] += probes
        self.misses += 1
        _STATE_METRICS["template_builds"] += 1
        _STATE_METRICS["template_build_nodes"] += counter[0]
        ref = TemplateRef.from_validated(
            self._next_id, generation, fingerprint, snapshot,
            "", counter[0])
        self._next_id += 1
        try:
            self._live.add(ref)
        except TypeError:  # pragma: no cover - refs are weakref-able
            pass
        if ref.retained_bytes > self.retained_bytes_limit:
            # A single item cannot be retained without exceeding the caller's
            # remaining byte budget.  It is still a valid live ref; it is just
            # not admitted to the LRU/index.
            self.uncached += 1
            self.oversized += 1
            self.rejected += 1
            return ref
        self._refs[ref.template_id] = ref
        self._buckets.setdefault(ref.fingerprint, []).append(ref)
        self._touch(ref)
        self._retained_bytes += ref.retained_bytes
        self._evict_oldest()
        return ref

    def intern_engine_static(
        self,
        static: tuple,
        *,
        generation: Hashable = 0,
    ) -> TemplateRef:
        """Intern the known Engine static schema without rebuilding the tree.

        The schema is validated directly into an exact reusable key.  A hit
        returns the already fixed canonical tree and only the key/probe cost is
        charged; only a miss freezes the tree.
        """
        if not isinstance(static, tuple):
            raise PeriodicContractError("periodic_intern_template_must_be_tuple")
        if not isinstance(generation, (int, str)) or isinstance(generation, bool):
            raise PeriodicContractError(
                f"periodic_intern_generation_invalid:{generation!r}")
        if self._generation != generation:
            self.evict_generation(generation)
            self._generation = generation
        counter = [0]
        key = _engine_static_key(static, counter)
        fingerprint = hash(key)
        probes = 0
        for entry_key, ref in self._engine_buckets.get(fingerprint, ()):
            probes += 1
            if ref.generation == generation and entry_key == key:
                _STATE_METRICS["template_hits"] += 1
                _STATE_METRICS["template_hit_nodes"] += counter[0]
                _STATE_METRICS["template_probes"] += probes
                self.hits += 1
                self._touch(ref)
                return ref
        _STATE_METRICS["template_probes"] += probes
        self.misses += 1
        build_counter = [0]
        snapshot = _freeze_snapshot(
            static, path="template", counter=build_counter)
        _STATE_METRICS["template_builds"] += 1
        _STATE_METRICS["template_build_nodes"] += build_counter[0]
        ref = TemplateRef._from_internal_frozen(
            self._next_id, generation, fingerprint, snapshot, build_counter[0],
        )
        self._next_id += 1
        try:
            self._live.add(ref)
        except TypeError:  # pragma: no cover - refs are weakref-able
            pass
        if ref.retained_bytes > self.retained_bytes_limit:
            self.uncached += 1
            self.oversized += 1
            self.rejected += 1
            return ref
        self._refs[ref.template_id] = ref
        self._engine_buckets.setdefault(
            ref.fingerprint, []).append((key, ref))
        self._touch(ref)
        self._retained_bytes += ref.retained_bytes
        self._evict_oldest()
        return ref

    def stats(self) -> dict:
        live = [ref for ref in self._live if ref is not None]
        return {
            "templates": self.templates,
            "limit": self.limit,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "retained_bytes": self._retained_bytes,
            "retained_bytes_limit": self.retained_bytes_limit,
            "uncached": self.uncached,
            "oversized": self.oversized,
            "rejected": self.rejected,
            "live_refs": len(live),
            "live_bytes": sum(ref.retained_bytes for ref in live),
        }


@dataclass(frozen=True)
class CanonicalState:
    """Validated invariant template plus only the changed dynamic residue.

    ``template`` is already validated and interned; ``dynamic`` is validated
    here on every construction (the changed restricted fields).  A dynamic
    fingerprint is a fast-reject only; exact ``dynamic`` equality remains the
    match condition.

    ``retained_bytes`` is the real ``sys.getsizeof`` total of the frozen
    dynamic tree.  It is measured while that tree is built, so a history frame
    can charge its own evidence in O(1) instead of re-walking the whole frozen
    graph.  It stays 0 for the generic validated path, whose cost is measured
    by ``_measure_retained`` at admission time.
    """

    template: TemplateRef
    dynamic: tuple
    version: int = 0
    origin: str = "trusted"
    dynamic_fingerprint: int | None = None
    nodes: int = 0
    retained_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.template, TemplateRef):
            raise PeriodicContractError("periodic_canonical_template_invalid")
        if not isinstance(self.dynamic, tuple):
            raise PeriodicContractError("periodic_canonical_dynamic_must_be_tuple")
        counter = [0]
        sizes = [0] if self.origin == "engine" else None
        if self.origin == "engine":
            snapshot = _freeze_engine_dynamic(self.dynamic, counter, sizes)
        else:
            _check_structural(self.dynamic, path="dynamic", counter=counter)
            # Own the observed value: a later caller mutation of a shared
            # list/dict must not rewrite this historical frontier.
            snapshot = _freeze_snapshot(
                self.dynamic, path="dynamic", counter=counter)
        object.__setattr__(self, "dynamic", snapshot)
        object.__setattr__(self, "dynamic_fingerprint", hash(snapshot))
        object.__setattr__(self, "nodes", counter[0])
        if sizes is not None:
            object.__setattr__(self, "retained_bytes", int(sizes[0]))
        _require_int("canonical_version", self.version)
        if self.origin not in {"engine", "trusted", "external"}:
            raise PeriodicContractError(f"periodic_canonical_origin_invalid:{self.origin!r}")
        _STATE_METRICS["dynamic_validations"] += 1
        _STATE_METRICS["dynamic_nodes"] += counter[0]


_DEFAULT_INTERNER = StateInterner(
    limit=_DEFAULT_INTERNER_LIMIT,
    retained_bytes_limit=_DEFAULT_INTERNER_BYTES,
)


def reset_state_caches() -> None:
    """Reset the process-local template interner and structural counters."""
    _DEFAULT_INTERNER.clear()
    reset_metrics()


_ENGINE_DYNAMIC_ARITY = 5
_ENGINE_TASK_DYNAMIC_ARITY = 4


def _frozen_scalar_close(
    left: tuple,
    right: tuple,
    *,
    clock_scale: float = 0.0,
    rel_tol: float = 0.0,
) -> bool:
    if left == right:
        return True
    if (
        len(left) == 2
        and len(right) == 2
        and left[0] == "float"
        and right[0] == "float"
    ):
        return _numeric_close(
            left[1], right[1], clock_scale=clock_scale, rel_tol=rel_tol,
        )
    return False


def _engine_dynamic_close(
    first: tuple,
    second: tuple,
    *,
    clock_scale: float,
    span: float,
) -> bool:
    """Compare frozen ``origin=='engine'`` dynamic trees allowing clock-scale ULP jitter."""
    if (
        len(first) != 2
        or len(second) != 2
        or first[0] != "tuple"
        or second[0] != "tuple"
    ):
        return False
    f_body, s_body = first[1], second[1]
    if len(f_body) != _ENGINE_DYNAMIC_ARITY or len(s_body) != _ENGINE_DYNAMIC_ARITY:
        return False
    # runs (1), allocations (2), pools (3) stay 100% bit-exact.
    if f_body[1] != s_body[1] or f_body[2] != s_body[2] or f_body[3] != s_body[3]:
        return False
    # task_dynamic (0): ("tuple", (task_entry, ...))
    f_tasks, s_tasks = f_body[0], s_body[0]
    if f_tasks != s_tasks:
        if f_tasks[0] != "tuple" or s_tasks[0] != "tuple" or len(f_tasks[1]) != len(s_tasks[1]):
            return False
        for ft, st in zip(f_tasks[1], s_tasks[1]):
            if ft == st:
                continue
            f_fields, s_fields = ft[1], st[1]
            if (
                len(f_fields) != _ENGINE_TASK_DYNAMIC_ARITY
                or len(s_fields) != _ENGINE_TASK_DYNAMIC_ARITY
            ):
                return False
            # remaining (0) and waiting (1) stay exact so a live progressing
            # residual task is never certified as invariant during a skip.
            if f_fields[0] != s_fields[0] or f_fields[1] != s_fields[1]:
                return False
            # release (2) and start (3) offsets are time deltas in ms
            if not _frozen_scalar_close(f_fields[2], s_fields[2], clock_scale=clock_scale):
                return False
            if not _frozen_scalar_close(f_fields[3], s_fields[3], clock_scale=clock_scale):
                return False
    # releases (4): ("tuple", (("tuple", ((tag, when_offset), key)), ...))
    f_rel, s_rel = f_body[4], s_body[4]
    if f_rel != s_rel:
        if f_rel[0] != "tuple" or s_rel[0] != "tuple" or len(f_rel[1]) != len(s_rel[1]):
            return False
        for fe, se in zip(f_rel[1], s_rel[1]):
            if fe == se:
                continue
            if len(fe[1]) != 2 or len(se[1]) != 2 or fe[1][1] != se[1][1]:
                return False
            if not _frozen_scalar_close(fe[1][0], se[1][0], clock_scale=clock_scale):
                return False
    return True


def _canonical_equal(
    first: CanonicalState,
    second: CanonicalState,
    *,
    clock_scale: float = 0.0,
    span: float = 0.0,
) -> bool:
    """Exact canonical comparison with clock-scale ULP tolerance for Engine dynamic floats."""
    if first.origin == "external" or second.origin == "external":
        # External canonical values keep the full exact comparison path; only
        # the trusted engine/trusted origins may use the template identity.
        return _structural_equal((first.template.template, first.dynamic), (second.template.template, second.dynamic))
    if (
        first.template.fingerprint != second.template.fingerprint
        or first.template.generation != second.template.generation
    ):
        return False
    if first.template is not second.template and first.template.template != second.template.template:
        return False
    if (
        first.dynamic_fingerprint is not None
        and second.dynamic_fingerprint is not None
        and first.dynamic_fingerprint == second.dynamic_fingerprint
        and first.dynamic == second.dynamic
    ):
        return True
    if first.origin == "engine" and second.origin == "engine" and clock_scale > 0.0:
        return _engine_dynamic_close(
            first.dynamic, second.dynamic, clock_scale=clock_scale, span=span,
        )
    if (
        first.dynamic_fingerprint is not None
        and second.dynamic_fingerprint is not None
        and first.dynamic_fingerprint != second.dynamic_fingerprint
    ):
        return False
    return first.dynamic == second.dynamic
