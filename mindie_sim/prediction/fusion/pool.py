"""Private recipe-book overlays and real shared-pool prefetch programs.

[Role] Clone a ``RecipeBook`` so a trial owns its registry, register one
  occurrence-scoped recipe, and run the prefetch/release pair that acquires the
  declared L1 pool at copy start and frees it only after both the producer copy
  and the consumer have retired.
[Invariants] A rejected trial can never mutate the baseline or an accepted
  book. Pool grants are real ``Engine`` allocations with declared units; a
  reservation is held from prefetch start until both the producer copy and the
  consumer retire. Each running engine owns a lease keyed by (occurrence id,
  invocation token, physical pool scope, replica, stage), so interleaved DP/PP
  replicas and overlapping same-scope invocations acquire and release their own
  exact per-unit grants; a release never pops another invocation's handle.
  Every acquisition mints a fresh monotonic generation id, even for an explicit
  caller ``lease_token``.  A live key stays reserved while its free is merely
  scheduled; the key is retired only when the Engine free task actually
  executes.  Copy handles and the token label's first binding are persistent
  capabilities, so a stale or double handle release is a harmless no-op and can
  never free a newer generation.  An explicit or registered copy-event handle
  always outranks a reusable token alias; a token-only release consumes that
  label's oldest unclaimed live generation. A new acquisition is rejected
  while that token has an unclaimed live generation; failed release submission
  restores the original generation's place in that queue.
[Timeouts & Locks] Finite event/program construction; no I/O, clocks or locks.
[Side-Effect Policy] Pure book construction plus Engine submissions supplied by
  the caller's recipe context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import weakref
from typing import Any, Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.execution.weight_prefetch import PREFETCH_TASK_PRIORITY
from mindie_sim.prediction.program import RecipeBook


def clone_book(book: RecipeBook) -> RecipeBook:
    """A private overlay: dictionaries/frozensets are copied, recipes are shared."""
    clone = object.__new__(RecipeBook)
    clone.__dict__.update(book.__dict__)
    clone.recipes = dict(book.recipes)
    clone.families = dict(book.families)
    clone.programs = dict(book.programs)
    clone._explicit_recipes = frozenset(book._explicit_recipes)
    clone._caller_recipes = frozenset(book._caller_recipes)
    clone._explicit_families = frozenset(book._explicit_families)
    clone._family_kinds = dict(getattr(book, "_family_kinds", {}))
    return clone


def register_recipe(book: RecipeBook, recipe_id: str, recipe) -> RecipeBook:
    """Return a clone with one recipe registered; never edit the input book."""
    clone = clone_book(book)
    clone.recipes[recipe_id] = recipe
    clone._explicit_recipes = frozenset(set(clone._explicit_recipes) | {recipe_id})
    clone._caller_recipes = frozenset(set(clone._caller_recipes) | {recipe_id})
    return clone


_POOL_SCOPE = re.compile(r"(?:^|@)r(?P<replica>\d+):s(?P<stage>\d+)$")
_RUNTIME_PREFIX = re.compile(r"^r(?P<replica>\d+):i(?P<iteration>\d+):")
_RELEASE_SUFFIX = ":release"
_CANONICAL_COPY_PREFIX = "prefetch_W_"
_CANONICAL_RELEASE_PREFIX = "release_W_"


def lease_identity(scope: str, name: str = "") -> tuple:
    """Exact physical lease identity: pool scope + replica/stage + iteration.

    Canonical programs run several replicas/stages on one shared ``Engine``;
    the per-(replica, stage) pool namespace alone already distinguishes their
    buffer accounting, but a repeated iteration of one replica must not
    overwrite an earlier grant either.  Both the copy and the release task
    names carry the same ``rN:iM:`` submission prefix, so that prefix is part
    of the identity when present.
    """
    text = str(scope or "")
    match = _POOL_SCOPE.search(text)
    replica = int(match.group("replica")) if match is not None else None
    stage = int(match.group("stage")) if match is not None else None
    runtime = _RUNTIME_PREFIX.match(str(name or ""))
    iteration = int(runtime.group("iteration")) if runtime is not None else None
    return (text, replica, stage, iteration)


def _without_runtime_prefix(name: str) -> str:
    return _RUNTIME_PREFIX.sub("", str(name or ""), count=1)


def invocation_token(name: str = "") -> str:
    """Verbatim caller invocation key shared by one copy task and its release.

    Only the execution-scoped ``rN:iM:`` prefix and the canonical
    ``prefetch_W_`` label are structural; every caller-supplied suffix is part
    of the invocation name and is never stripped.  ``trial`` and
    ``trial:copy`` therefore remain two different leases.
    """
    text = _without_runtime_prefix(name)
    if text.startswith(_CANONICAL_COPY_PREFIX):
        text = text[len(_CANONICAL_COPY_PREFIX):]
    elif text.startswith(_CANONICAL_RELEASE_PREFIX):
        text = text[len(_CANONICAL_RELEASE_PREFIX):]
    return text or "lease"


def release_invocation_token(name: str = "") -> str:
    """Target invocation key for a release name.

    Exactly one trailing ``:release`` role suffix is removed (never a repeated
    suffix chain), so ``trial:release`` maps to ``trial`` and
    ``trial:copy:release`` maps to ``trial:copy``: an exact bijection.  The
    canonical ``release_W_<op>`` label maps to the same ``<op>`` key its
    ``prefetch_W_<op>`` copy used.
    """
    text = _without_runtime_prefix(name)
    if text.endswith(_RELEASE_SUFFIX):
        text = text[:-len(_RELEASE_SUFFIX)]
    if text.startswith(_CANONICAL_RELEASE_PREFIX):
        text = text[len(_CANONICAL_RELEASE_PREFIX):]
    elif text.startswith(_CANONICAL_COPY_PREFIX):
        text = text[len(_CANONICAL_COPY_PREFIX):]
    return text or "lease"


@dataclass
class _Grant:
    """One generation-scoped pool capability and its true lifetime."""
    grant_id: str
    key: tuple
    allocation: Allocation
    release_scheduled: bool = False
    retired: bool = False
    token: str | None = None


class _Grants:
    """One engine's live keys plus persistent capability history.

    ``active_by_key`` contains only grants whose Engine allocation is still
    live.  ``by_id`` and ``capabilities`` are never pruned: a stale copy handle
    (or the token label's first binding) keeps its original generation and
    becomes a no-op instead of falling back to a newer textual-key lease.
    ``token_live`` orders each reusable token label's unclaimed live
    generations so a token-only release has one deterministic target.
    """

    __slots__ = ("active_by_key", "by_id", "capabilities", "copy_handles",
                 "token_live")

    def __init__(self):
        self.active_by_key: dict[tuple, _Grant] = {}
        self.by_id: dict[str, _Grant] = {}
        self.capabilities: dict[tuple[str, object], str] = {}
        # Every copy event sequence is recorded, even a zero-byte copy that
        # acquired no grant, so a release cannot mistake it for a plain wait
        # dependency and fall back to a textual key.
        self.copy_handles: set[int] = set()
        self.token_live: dict[str, list[str]] = {}


class PoolHolder:
    """Per-Engine lease registry shared by one copy/release recipe pair.

    Recipes are shared between cloned books and may run in several concurrent
    ``Engine`` evaluations.  Allocation state therefore cannot live in one
    mutable slot: every engine owns its own leases keyed by this holder's
    occurrence-scoped lease id, the exact ``(replica, stage, iteration)``
    physical submission identity, and the caller's verbatim invocation name.
    Each acquire mints a strictly monotonic generation id and binds it to the
    returned copy event, so a release can free exactly the grant it was given
    even when several invocations overlap.  A live key is never a list: a
    second overlapping acquire under the same key is rejected before
    submission, including while its free is scheduled but not yet executed.  A
    key may be reused only after the Engine free task has actually retired its
    grant.  Weak keys make a destroyed engine's lease unreachable even if it
    never reached release.
    """

    def __init__(self, lease_id: str = "lease"):
        self.lease_id = str(lease_id)
        self._leases: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._fallback: dict[int, tuple[object, _Grants]] = {}
        self.generation = 0

    def _bucket(self, engine: Any) -> _Grants:
        try:
            grants = self._leases.get(engine)
            if grants is None:
                grants = _Grants()
                self._leases[engine] = grants
            return grants
        except TypeError:  # pragma: no cover - Engine is weak-referenceable
            entry = self._fallback.get(id(engine))
            if entry is None or entry[0] is not engine:
                entry = (engine, _Grants())
                self._fallback[id(engine)] = entry
            return entry[1]

    @staticmethod
    def _key_text(key: tuple) -> str:
        return ":".join(str(part) for part in key)

    def allocate(self, engine: Any, allocation: Allocation, *,
                 scope: str = "", name: str = "",
                 token: str | None = None) -> str:
        """Register one live grant; reject an overlapping live key outright."""
        invocation = invocation_token(name) if token is None else str(token)
        key = (self.lease_id, invocation, *lease_identity(scope, name))
        grants = self._bucket(engine)
        if key in grants.active_by_key:
            # Whether the release is pending or not, the Engine allocation is
            # still live until the free task executes at ``t_free``.
            raise ValueError("pool_lease_key_already_live:" + self._key_text(key))
        if token is not None and self._token_grants(grants, str(token)):
            # A mutable caller alias cannot identify two concurrently live
            # generations: a later token-only release would be ambiguous.
            raise ValueError("pool_lease_token_already_live:" + str(token))
        # A caller token is a capability alias, never the generation identity.
        # Every acquisition, including one that reuses a token, mints a fresh
        # strictly monotonic id; the handle remains the exact capability.
        self.generation += 1
        grant_id = f"grant:{self.generation}"
        grant = _Grant(grant_id, key, allocation,
                       token=str(token) if token is not None else None)
        grants.active_by_key[key] = grant
        grants.by_id[grant_id] = grant
        if token is not None:
            # The token alias is a caller label, never the mutable target of a
            # release: it keeps the first generation it ever named, while the
            # ordered ``token_live`` queue tracks later unclaimed generations.
            grants.capabilities.setdefault(("token", str(token)), grant_id)
            grants.token_live.setdefault(str(token), []).append(grant_id)
        return grant_id

    @staticmethod
    def _token_grants(grants: _Grants, token: str) -> tuple[str, ...]:
        """Unclaimed, still-live generations of one token, oldest first."""
        live = []
        for grant_id in grants.token_live.get(token, ()):
            grant = grants.by_id.get(grant_id)
            if grant is None or grant.retired or grant.release_scheduled:
                continue
            if grants.active_by_key.get(grant.key) is not grant:
                continue
            live.append(grant_id)
        return tuple(live)

    @staticmethod
    def _forget_token(grants: _Grants, grant: _Grant) -> None:
        """Retire one generation from its token's unclaimed FIFO."""
        if grant.token is None:
            return
        remaining = [grant_id for grant_id
                     in grants.token_live.get(grant.token, ())
                     if grant_id != grant.grant_id]
        if remaining:
            grants.token_live[grant.token] = remaining
        else:
            grants.token_live.pop(grant.token, None)

    def bind_handle(self, engine: Any, event: Any, grant: str | None) -> None:
        """Mark the actual copy event as a persistent capability handle."""
        sequence = getattr(event, "sequence", None)
        if sequence is None:
            return
        grants = self._bucket(engine)
        grants.copy_handles.add(int(sequence))
        if grant is None:
            return
        # Persistent capability history: retirement never erases this binding.
        grants.capabilities[("handle", int(sequence))] = str(grant)

    @staticmethod
    def _known_copy_handle(grants: _Grants, handle: Any) -> bool:
        sequence = getattr(handle, "sequence", None)
        return sequence is not None and int(sequence) in grants.copy_handles

    def claim_release(self, engine: Any, *, scope: str = "",
                      name: str = "", token: str | None = None,
                      handle: Any = None, strict_handle: bool = False
                      ) -> _Grant | None:
        """Reserve the exact grant named by capability, handle or release key.

        Resolution precedence is immutable capability first: an explicitly
        supplied ``copy_handle`` (even an unknown one) or a registered copy
        event selects exactly that generation, and a retired handle is a no-op
        that can never fall through to a newer mutable token alias.  A pure
        token-only release consumes that token's oldest unclaimed live
        generation.  A textual-key-only release may claim the currently active
        grant once; a second free is a no-op even if a newer generation later
        reuses the key.
        """
        grants = self._bucket(engine)
        grant: _Grant | None = None
        explicit_handle = handle is not None and (
            strict_handle or self._known_copy_handle(grants, handle))
        if explicit_handle:
            sequence = getattr(handle, "sequence", None)
            if sequence is not None:
                grant_id = grants.capabilities.get(("handle", int(sequence)))
                if grant_id is not None:
                    grant = grants.by_id.get(grant_id)
        elif token is not None:
            token_ids = self._token_grants(grants, str(token))
            if token_ids:
                grant = grants.by_id.get(token_ids[0])
        else:
            invocation = release_invocation_token(name)
            key = (self.lease_id, invocation, *lease_identity(scope, name))
            grant = grants.active_by_key.get(key)
        if grant is None or grant.retired or grant.release_scheduled:
            return None
        if grants.active_by_key.get(grant.key) is not grant:
            return None
        grant.release_scheduled = True
        # The claimed generation is no longer available to a later token-only
        # release; the persistent ``by_id``/``capabilities`` history remains.
        self._forget_token(grants, grant)
        return grant

    def rollback_release(self, engine: Any, grant: _Grant) -> None:
        """Restore one uncommitted claim to its monotonic token FIFO position."""
        grants = self._bucket(engine)
        if grants.by_id.get(grant.grant_id) is not grant:
            return
        grant.release_scheduled = False
        if (grant.token is None or grant.retired
                or grants.active_by_key.get(grant.key) is not grant):
            return
        queue = grants.token_live.setdefault(grant.token, [])
        if grant.grant_id in queue:
            return
        generation = int(grant.grant_id.split(":")[1])
        for index, grant_id in enumerate(queue):
            if int(grant_id.split(":")[1]) > generation:
                queue.insert(index, grant.grant_id)
                break
        else:
            queue.append(grant.grant_id)

    def finish_release(self, engine: Any, grant_id: str) -> None:
        """Retire a grant only when its Engine free task actually executed."""
        grants = self._bucket(engine)
        grant = grants.by_id.get(str(grant_id))
        if grant is None:
            return
        if grants.active_by_key.get(grant.key) is grant:
            grants.active_by_key.pop(grant.key, None)
        grant.retired = True
        self._forget_token(grants, grant)

    def retire(self, engine: Any, allocation: Allocation) -> bool:
        """Remove exactly one handle (failed-submit rollback), never a peer's."""
        grants = self._bucket(engine)
        for key, grant in list(grants.active_by_key.items()):
            if grant.allocation is allocation:
                grants.active_by_key.pop(key, None)
                grant.retired = True
                self._forget_token(grants, grant)
                return True
        return False


def pool_units(units: int) -> tuple[int, ...]:
    return tuple(range(max(1, int(units))))


def units_for_pool(context, pool: str = "l1") -> int:
    declared = int(dict(getattr(context, "pool_units", {}) or {}).get(pool, 1) or 1)
    return max(1, declared)


class PrefetchPoolProgram:
    """Copy work plus a real L1 grant acquired at task start."""

    def __init__(self, holder: PoolHolder, amount: float, units: int,
                 *, caps: Mapping[str, float] | None = None, priority: int = PREFETCH_TASK_PRIORITY,
                 work_resource: str = "hbm", pool_name: str = "l1",
                 lease_token: str | None = None):
        self.holder = holder
        self.amount = float(amount)
        self.units = max(1, int(units))
        self.caps = dict(caps or {})
        self.priority = int(priority)
        self.work_resource = str(work_resource)
        self.pool_name = str(pool_name)
        self.lease_token = lease_token

    def start(self, engine, name, ready, *, owner=""):
        if str(name).endswith(_RELEASE_SUFFIX):
            # The copy and release namespaces are disjoint: a copy named
            # ``x:release`` could not be paired unambiguously with ``x``.
            raise ValueError(f"prefetch_invocation_name_reserved:{name}")
        per_unit = (self.amount / self.units) if self.amount > 0 else 0.0
        allocation = (Allocation(f"{name}:pool", self.pool_name, per_unit,
                                 pool_units(self.units))
                      if per_unit > 0 else None)
        grant = None
        if allocation is not None:
            grant = self.holder.allocate(
                engine, allocation, scope=self.pool_name, name=name,
                token=self.lease_token)
        committed = False
        try:
            event = engine.submit(
                name + ":copy",
                {self.work_resource: self.amount} if self.amount else {},
                after=(ready,), caps=self.caps, priority=self.priority,
                owner=owner,
                acquire=(allocation,) if allocation is not None else ())
            committed = True
        finally:
            if not committed and allocation is not None:
                self.holder.retire(engine, allocation)
        # Mark every copy event, including a zero-byte copy that acquired no
        # grant, so its handle is an unambiguous capability.
        self.holder.bind_handle(engine, event, grant)
        return event


class ReleasePoolProgram:
    """Zero-work free of the exact grant acquired by the matching copy program."""

    def __init__(self, holder: PoolHolder, scope: str = "l1",
                 lease_token: str | None = None):
        self.holder = holder
        self.scope = str(scope)
        self.lease_token = lease_token

    def start(self, engine, name, ready, *, owner="", copy_handle=None):
        handle = ready if copy_handle is None else copy_handle
        grant = self.holder.claim_release(
            engine, scope=self.scope, name=name, token=self.lease_token,
            handle=handle, strict_handle=copy_handle is not None)
        if grant is None:
            # Unknown, retired or already-scheduled capability: never touch a
            # newer textual-key generation.
            return engine.submit(name + ":noop", after=(ready,), owner=owner)
        committed = False
        try:
            event = engine.submit(name + ":free", after=(ready,),
                                  free=(grant.allocation,), owner=owner)
            committed = True
        finally:
            if not committed:
                self.holder.rollback_release(engine, grant)
        # The live key remains reserved until the Engine free task actually
        # executes at ``t_free`` and the pool bytes are returned.
        engine.on_complete(
            event,
            lambda _event: self.holder.finish_release(engine, grant.grant_id))
        return event


def declared_l1_holders(plan) -> int:
    """L1 bytes declared by selected device-kernel groups in the current plan."""
    total = 0
    for op in getattr(plan, "ops", ()):
        metadata = dict(op.metadata)
        info = metadata.get("device_kernel_group")
        if isinstance(info, Mapping):
            for buffer in info.get("buffers", ()) or ():
                if not isinstance(buffer, Mapping) or buffer.get("pool") != "l1":
                    continue
                units = buffer.get("units") or (0,)
                total += int(buffer.get("bytes", 0) or 0) * len(tuple(units))
        hold = metadata.get("optimizer_l1_hold")
        if isinstance(hold, Mapping):
            total += int(hold.get("bytes", 0) or 0) * len(
                tuple(hold.get("units") or (0,)))
    return int(total)


def l1_peak_holders(trace, start: float, end: float) -> int:
    """Peak live L1 bytes in ``[start, end]`` from real Engine grant/release events."""
    if not trace:
        return 0
    events = sorted(
        (float(row.get("time", 0.0)), row.get("action"), int(row.get("size", 0)),
         tuple(row.get("units") or ()))
        for row in trace if str(row.get("pool", "")).startswith("l1"))
    live = 0
    peak = 0
    for time, action, size, units in events:
        if time > end + 1e-12:
            break
        delta = size * len(units)
        if action == "acquire":
            live += delta
        elif action == "release":
            live = max(0, live - delta)
        if time >= start - 1e-12:
            peak = max(peak, live)
    return int(peak)


@dataclass
class PoolLedger:
    """Residency intervals of prefetched tiles: start -> consumer completion."""
    intervals: list = field(default_factory=list)

    def add(self, start: float, end: float, amount: int) -> None:
        if amount > 0 and end >= start:
            self.intervals.append((float(start), float(max(start, end)), int(amount)))

    @property
    def reserved(self) -> int:
        return sum(amount for _, _, amount in self.intervals)

    def peak(self) -> int:
        events: list[tuple[float, int, int]] = []
        for start, end, amount in self.intervals:
            events.append((start, 0, amount))
            events.append((max(start, end), 1, amount))
        live = peak = 0
        for _time, kind, amount in sorted(events):
            live += amount if kind == 0 else -amount
            peak = max(peak, live)
        return peak
