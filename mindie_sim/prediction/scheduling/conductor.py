"""Fragment conductor on the shared Engine.

[Role] Bind fragment resources, track event generations and aggregate results.
[Invariants] Event publication and allocation planning are validated before admission.
[Timeouts & Locks] Event, stream and observation registries have explicit bounds.
[Side-Effect Policy] Updates conductor registries and submits to the caller's Engine.
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from typing import Any, Mapping, Sequence

from mindie_sim.schema.operators import ProgramFragment, StateEffect

from ..engine import Allocation, Engine, Event
from .allocations import _AllocationPlan, _plan_allocations
from .state import (
    _DEFAULT_STATE_VERSION_LIMIT,
    _MISSING,
    _STATE_ACTIONS_WRITE,
    FragmentContractError,
    UnsupportedFragmentSemantics,
    _nonnegative_int,
)
from .submission import FragmentHandle, _LiveRecord, _submit_fragment

_DEFAULT_FRAGMENT_CACHE_LIMIT = 64
_DEFAULT_FRAGMENT_EVENT_LIMIT = 4096


class FragmentConductor:
    """Submit fragments into one shared Engine and bounded aggregate their results."""

    def __init__(
        self, engine: Engine, *,
        resource_bindings: Mapping[str, str] | None = None,
        pool_bindings: Mapping[str, str] | None = None,
        unit_conversions: Mapping[tuple[str, str], float] | None = None,
        state: Mapping[str, int] | None = None,
        state_version_limit: int = _DEFAULT_STATE_VERSION_LIMIT,
        detail: bool = False, cache_limit: int = _DEFAULT_FRAGMENT_CACHE_LIMIT,
        repeat_limit: int = 1_000_000, stream_limit: int = 1024,
        event_limit: int = _DEFAULT_FRAGMENT_EVENT_LIMIT, prefix: str = "fragment:",
        memory=None, memory_replica: int = 0, memory_stage: int = 0,
        memory_pools: Sequence[str] = ("hbm",),
        memory_owner=None, memory_storage=None, memory_ranges=None,
        memory_generation=None, memory_event=None, memory_host_bytes: float = 0.0,
        memory_unit: str | None = None,
        memory_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        if not isinstance(engine, Engine):
            raise FragmentContractError("fragment_conductor_requires_engine")
        self.engine = engine
        self.memory = memory if memory is not None else engine.memory
        self.memory_replica = int(memory_replica)
        self.memory_stage = int(memory_stage)
        self.memory_pools = frozenset(str(item) for item in memory_pools)
        self.memory_owner = memory_owner
        self.memory_storage = memory_storage
        self.memory_ranges = memory_ranges
        self.memory_generation = memory_generation
        self.memory_event = memory_event
        self.memory_host_bytes = float(memory_host_bytes)
        self.memory_unit = memory_unit
        if memory is not None and engine.memory is not None and engine.memory is not memory:
            # PD endpoints can share one Engine with different ledgers. Each
            # task carries its own ledger; a second conductor must not rewrite
            # the first endpoint's default pool set.
            pass
        elif memory is not None:
            engine.bind_memory(memory, pools=self.memory_pools)
        self.memory_bindings = {
            str(name): dict(binding)
            for name, binding in dict(memory_bindings or {}).items()}
        self.resource_bindings = {str(key): str(value)
                                  for key, value in dict(resource_bindings or {}).items()}
        self.pool_bindings = {str(key): str(value)
                              for key, value in dict(pool_bindings or {}).items()}
        self.unit_conversions: dict[tuple[str, str], float] = {}
        for key, factor in dict(unit_conversions or {}).items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or not all(isinstance(item, str) and item for item in key)):
                raise FragmentContractError(
                    f"fragment_unit_conversion_key_invalid:{key!r}")
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) \
                    or not math.isfinite(float(factor)) or float(factor) <= 0:
                raise FragmentContractError(
                    f"fragment_unit_conversion_factor_invalid:{key}")
            self.unit_conversions[(key[0], key[1])] = float(factor)
        # The typed state ledger belongs to the shared RuntimeState. An
        # explicitly supplied StateLedger is adopted as-is; a plain mapping is
        # an initial-version seed for that one runtime-owned ledger. A
        # conductor never owns a second, independent version map.
        self.state = engine.ensure_state_ledger(
            state, version_limit=state_version_limit)
        self.detail = bool(detail)
        self.cache_limit = max(1, _nonnegative_int("fragment_cache_limit", cache_limit))
        self.repeat_limit = max(1, _nonnegative_int("fragment_repeat_limit", repeat_limit))
        self.stream_limit = max(1, _nonnegative_int("fragment_stream_limit", stream_limit))
        self.event_limit = max(1, _nonnegative_int("fragment_event_limit", event_limit))
        self.prefix = str(prefix)
        self._streams: dict[str, Event] = {}
        self._events: OrderedDict[tuple[str, int], Event] = OrderedDict()
        self._events_evicted = 0
        self._event_evicted_names: set[str] = set()
        self._event_history_ambiguous = False
        self._generations: dict[str, int] = {}
        self._live: OrderedDict[int, _LiveRecord] = OrderedDict()
        self._recent: deque[dict[str, Any]] | None = (
            deque() if self.detail else None)
        self._recent_dropped = 0
        self._aggregates: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._aggregates_evicted = 0
        self._unsupported: OrderedDict[str, int] = OrderedDict()
        self._unsupported_overflow = 0
        self._sequence = 0
        self._totals = {
            "fragments": 0, "repeats": 0, "batched": 0, "expanded": 0,
            "release_events": 0,
        }
        self._total_service: dict[str, float] = {}

    # -- public surface -------------------------------------------------
    def check(self, fragment: ProgramFragment, *, holds: Mapping[str, Any] | None = None
              ) -> tuple[str, ...]:
        """Return unsupported executable-semantics reasons without submitting."""
        if not isinstance(fragment, ProgramFragment):
            raise FragmentContractError("fragment_requires_program_fragment")
        reasons: list[str] = []
        for effect in fragment.allocations:
            if effect.lifetime == "event":
                reasons.append(f"allocation_scope_unsupported:event:{effect.name}")
        if fragment.repeats > 1 and any(
                effect.action in _STATE_ACTIONS_WRITE
                for effect in fragment.state_effects):
            reasons.append("repeat_state_writes_unsupported")
        if fragment.repeats > 1 and holds:
            reasons.append("repeat_external_holds_unsupported")
        used = set(fragment.service_work)
        priorities = {policy.priority for resource, policy in fragment.policies
                      if resource in used and policy.priority}
        if len(priorities) > 1:
            reasons.append("fragment_priority_ambiguous")
        elif priorities and fragment.priority and fragment.priority not in priorities:
            reasons.append(
                f"fragment_priority_conflict:{fragment.priority}:{sorted(priorities)}")
        reasons = list(dict.fromkeys(reasons))
        for reason in reasons:
            self._record_unsupported(reason)
        return tuple(reasons)

    def _unit_factor(self, declared_unit: str, physical: str,
                     declared_resource: str) -> float:
        expected = self.engine.resources[physical].unit
        if declared_unit == expected:
            return 1.0
        factor = self.unit_conversions.get((declared_unit, expected))
        if factor is None:
            raise FragmentContractError(
                f"fragment_work_unit_mismatch:{declared_resource}:"
                f"{declared_unit}:{expected}")
        return factor

    def _validate_binding(self, fragment: ProgramFragment) -> None:
        declared = {name: policy for name, policy in fragment.policies}
        for item in fragment.work.items:
            physical = self._resource(item.resource)
            resource = self.engine.resources[physical]
            policy = declared.get(item.resource)
            if policy is not None and policy.sharing != resource.policy:
                raise FragmentContractError(
                    f"fragment_resource_policy_mismatch:{item.resource}:"
                    f"{physical}:{policy.sharing}:{resource.policy}")
            self._unit_factor(item.unit, physical, item.resource)

    def _batchable(self, fragment: ProgramFragment,
                   holds: Mapping[str, Any]) -> bool:
        if (fragment.repeats <= 1 or fragment.allocations
                or fragment.state_effects or holds):
            return False
        used = {self._resource(item.resource)
                for item in fragment.work.issued() if item.amount}
        if not used:
            return True
        if any(self.engine.resources[name].policy != "fair" for name in used):
            # FIFO arbitration and per-repeat start times are observable.
            return False
        # A single fair flow is fungible across repeats. Coupled resources
        # progress together, so one larger coupled flow is equivalent. A
        # non-coupled multi-resource repeat can finish one resource early and
        # release its share before the next repeat; batching would hide that.
        return len(used) == 1 or fragment.coupled

    def submit(
        self, fragment: ProgramFragment, *,
        after: Sequence[Event] = (), inputs: Mapping[str, Any] | None = None,
        conditions: Mapping[str, bool] | None = None,
        holds: Mapping[str, Any] | None = None,
        event_bindings: Mapping[str, Any] | None = None,
        owner: str = "",
    ) -> FragmentHandle:
        """Admit one fragment. Work advances only through ``engine.advance``."""
        return _submit_fragment(
            self, fragment, after=after, inputs=inputs, conditions=conditions,
            holds=holds, event_bindings=event_bindings, owner=owner)

    def advance(
        self, *, until: Sequence[Event] = (), horizon: float = math.inf,
        first: bool = False, max_steps: int | None = None,
    ) -> tuple[Event, ...]:
        """Advance the shared Engine; default targets all live fragments."""
        if not until:
            until = tuple(record.handle.event for record in self._live.values()
                          if record.handle is not None)
        return self.engine.advance(
            until=until, horizon=horizon, first=first, max_steps=max_steps)

    def event(self, name: str, generation: int | None = None) -> Event:
        """Look up one recorded event without guessing across generations."""
        event = self._find_event(name, generation)
        if event is None:
            raise FragmentContractError(
                f"fragment_event_unbound:{name}"
                + ("" if generation is None else f":{generation}"))
        return event

    def record_event(self, name: str, event: Event,
                     generation: int | None = None) -> int:
        """Publish a real recorded event for event-port consumers.

        The registry is an LRU of completed/recorded handles. Once it reaches
        ``event_limit`` the oldest record is reclaimed; a later consumer that
        asks for that exact generation receives an explicit unbound/generation
        error rather than a different same-named event.
        """
        if not isinstance(name, str) or not name:
            raise FragmentContractError("fragment_event_record_name_invalid")
        if not isinstance(event, Event):
            raise FragmentContractError(f"fragment_event_record_invalid:{name}")
        if generation is None:
            generation = self._generations.get(name, -1) + 1
        generation = _nonnegative_int("fragment_event_generation", generation)
        key = str(name), generation
        if key in self._events:
            raise FragmentContractError(
                f"fragment_event_generation_already_recorded:{name}:{generation}")
        # A name-limit rejection must not evict a healthy old record first.
        if name not in self._generations \
                and len(self._generations) >= self.event_limit:
            raise FragmentContractError(
                f"fragment_event_name_limit_exceeded:{self.event_limit}")
        while len(self._events) >= self.event_limit:
            evicted, _ = self._events.popitem(last=False)
            self._events_evicted += 1
            if len(self._event_evicted_names) < self.event_limit:
                self._event_evicted_names.add(evicted[0])
            else:
                # The per-name miss set itself must stay bounded. Once history
                # is ambiguous at this scale, require explicit generations for
                # every implicit lookup rather than guessing across old names.
                self._event_evicted_names.clear()
                self._event_history_ambiguous = True
        self._events[key] = event
        self._generations[str(name)] = max(
            generation, self._generations.get(str(name), 0))
        return generation

    def summary(self, detail: bool | None = None) -> dict[str, Any]:
        """Bounded aggregate; ``detail`` only retains recent observations."""
        if detail is None:
            detail = self.detail
        result = {
            "engine": self.engine.summary(),
            "live_fragments": len(self._live),
            "totals": dict(self._totals),
            "service_work": dict(sorted(self._total_service.items())),
            "fragments": {name: dict(value)
                          for name, value in self._aggregates.items()},
            "aggregates_evicted": self._aggregates_evicted,
            "unsupported": dict(self._unsupported),
            "unsupported_overflow": self._unsupported_overflow,
            "state": self.state.snapshot(),
            "streams": len(self._streams),
            "recorded_events": len(self._events),
            "events_evicted": self._events_evicted,
            "event_history_ambiguous": self._event_history_ambiguous,
            "recent": [],
        }
        if detail and self._recent is not None:
            result["recent"] = list(self._recent)
            result["recent_dropped"] = self._recent_dropped
        return result

    # -- lowering -------------------------------------------------------
    def _resource(self, declared: str) -> str:
        name = self.resource_bindings.get(str(declared), str(declared))
        if name not in self.engine.resources:
            raise FragmentContractError(
                f"fragment_resource_unbound:{declared}->{name}")
        return name

    def _pool(self, declared: str) -> str:
        name = self.pool_bindings.get(str(declared), str(declared))
        if name not in self.engine.pools:
            raise FragmentContractError(f"fragment_pool_unbound:{declared}->{name}")
        return name

    def _priority(self, fragment: ProgramFragment) -> int:
        used = set(fragment.service_work)
        priorities = {policy.priority for resource, policy in fragment.policies
                      if resource in used and policy.priority}
        if len(priorities) > 1:
            raise UnsupportedFragmentSemantics(
                f"fragment_priority_ambiguous:{fragment.name}")
        if priorities:
            value = next(iter(priorities))
            if fragment.priority and fragment.priority != value:
                raise UnsupportedFragmentSemantics(
                    f"fragment_priority_conflict:{fragment.priority}:{value}")
            return value
        return fragment.priority

    def _work(self, fragment: ProgramFragment, multiply: int = 1,
              *, kind: str = "issued") -> dict[str, float]:
        result: dict[str, float] = {}
        for item in fragment.work.items:
            if item.kind != kind or not item.amount:
                continue
            name = self._resource(item.resource)
            factor = self._unit_factor(item.unit, name, item.resource)
            try:
                scaled = (float(item.amount) / factor) * multiply
            except OverflowError as exc:
                raise FragmentContractError(
                    f"fragment_batched_work_overflow:{fragment.name}") from exc
            result[name] = result.get(name, 0.0) + scaled
        for value in result.values():
            if not math.isfinite(value):
                raise FragmentContractError(
                    f"fragment_batched_work_overflow:{fragment.name}")
        return result

    def _active(self, condition: str, conditions: Mapping[str, bool],
                owner: str) -> bool:
        if not condition:
            return True
        if condition not in conditions:
            raise FragmentContractError(
                f"conditional_effect_unbound:{owner}:{condition}")
        return bool(conditions[condition])

    def _active_effects(self, fragment: ProgramFragment,
                        conditions: Mapping[str, bool],
                        actions: set[str]) -> tuple[StateEffect, ...]:
        result = []
        for effect in fragment.state_effects:
            if effect.action not in actions:
                continue
            if self._active(effect.condition, conditions, effect.target):
                result.append(effect)
        return tuple(result)

    def _as_event(self, value: Any, label: str) -> Event:
        if isinstance(value, FragmentHandle):
            return value.event
        if isinstance(value, Event):
            return value
        raise FragmentContractError(f"fragment_event_binding_invalid:{label}")

    def _as_events(self, value: Any, label: str) -> tuple[Event, ...]:
        if isinstance(value, (FragmentHandle, Event)):
            return (self._as_event(value, label),)
        if isinstance(value, (tuple, list)):
            return tuple(self._as_event(item, label) for item in value)
        raise FragmentContractError(f"fragment_event_binding_invalid:{label}")

    def _port_events(self, value: Any, label: str) -> tuple[Event, ...]:
        if isinstance(value, (FragmentHandle, Event)):
            return (self._as_event(value, label),)
        ready = getattr(value, "ready", None)
        if isinstance(ready, (FragmentHandle, Event)):
            return (self._as_event(ready, label),)
        if isinstance(value, (tuple, list)):
            collected: list[Event] = []
            for item in value:
                item_events = self._port_events(item, label)
                if item_events:
                    collected.extend(item_events)
            return tuple(dict.fromkeys(collected))
        if isinstance(value, Mapping):
            if value.get("event") is not None:
                return self._as_events(value["event"], label)
            raw_ready = value.get("ready")
            if isinstance(raw_ready, (FragmentHandle, Event)):
                return (self._as_event(raw_ready, label),)
            if isinstance(raw_ready, (tuple, list)):
                ready_events = self._port_events(raw_ready, label)
                if ready_events:
                    return ready_events
        binding = getattr(value, "binding", "")
        binding_name = getattr(value, "binding_name", "")
        if binding == "event" and isinstance(binding_name, str) and binding_name:
            recorded = self._find_event(binding_name, None)
            if recorded is None:
                raise FragmentContractError(
                    f"fragment_event_unbound:{binding_name}")
            return (recorded,)
        return ()

    def _base_dependencies(self, fragment: ProgramFragment,
                           after: Sequence[Event],
                           inputs: Mapping[str, Any] | None) -> tuple[Event, ...]:
        provided = dict(inputs or {})
        events: list[Event] = [self._as_event(item, "after") for item in after]
        stream = fragment.occurrence.stream if fragment.occurrence else ""
        if stream and stream in self._streams:
            events.append(self._streams[stream])
        for group in ("tensor_in", "state_in", "event_in"):
            for port in getattr(fragment.ports, group):
                value = provided.get(port.name, _MISSING)
                if value is _MISSING:
                    if group == "event_in":
                        recorded = self._find_event(port.name, port.generation)
                        if recorded is not None:
                            events.append(recorded)
                            continue
                    if getattr(port, "optional", False):
                        continue
                    raise FragmentContractError(
                        f"fragment_input_unbound:{port.name}")
                if value is None:
                    raise FragmentContractError(
                        f"fragment_input_unbound:{port.name}")
                if group == "event_in":
                    events.extend(self._as_events(value, port.name))
                else:
                    events.extend(self._port_events(value, port.name))
        return tuple(dict.fromkeys(events))

    def _find_event(self, name: str, generation: int | None) -> Event | None:
        if generation is not None:
            return self._events.get((name, generation))
        if self._event_history_ambiguous or name in self._event_evicted_names:
            raise FragmentContractError(
                f"fragment_event_generation_required:{name}")
        matches = [event for (recorded, _), event in self._events.items()
                   if recorded == name]
        if len(matches) > 1:
            raise FragmentContractError(
                f"fragment_event_generation_required:{name}")
        return matches[0] if matches else None

    def hold_async_reader(self, request: str, consumer: str, *,
                          storage=None, ranges=None, generation=None,
                          event=None) -> bool:
        """Bind a real async reader before its submit; no synthetic hold."""
        if self.memory is None:
            raise FragmentContractError("fragment_memory_not_attached")
        return self.memory.hold_for_async_consumer(
            self.memory_replica, str(request), str(consumer),
            storage=storage if storage is not None else self.memory_storage,
            ranges=ranges if ranges is not None else self.memory_ranges,
            generation=(generation if generation is not None
                        else self.memory_generation),
            event=event if event is not None else self.memory_event)

    def release_async_reader(self, request: str, consumer: str) -> bool:
        if self.memory is None:
            raise FragmentContractError("fragment_memory_not_attached")
        return self.memory.release_async_consumer(
            self.memory_replica, str(request), str(consumer))

    def _plan_allocations(self, fragment: ProgramFragment,
                          conditions: Mapping[str, bool],
                          holds: Mapping[str, Any],
                          sequence: int) -> _AllocationPlan:
        return _plan_allocations(self, fragment, conditions, holds, sequence)

    def _allocation_binding(self, name: str) -> dict | None:
        """Merge the conductor default with one real allocation identity."""
        defaults = {
            "owner": self.memory_owner, "storage": self.memory_storage,
            "ranges": self.memory_ranges,
            "generation": self.memory_generation, "event": self.memory_event,
            "host_bytes": self.memory_host_bytes, "stage": self.memory_stage,
        }
        if self.memory_unit is not None:
            defaults["unit"] = self.memory_unit
        result = {key: value for key, value in defaults.items()
                  if value is not None}
        override = self.memory_bindings.get(name)
        if override is None:
            matches = [(len(str(key)), str(key))
                       for key in self.memory_bindings
                       if name.endswith(":" + str(key))
                       or name.endswith("#" + str(key))]
            if matches:
                matches.sort(reverse=True)
                if len(matches) > 1 and matches[0][0] == matches[1][0]:
                    raise FragmentContractError(
                        f"fragment_memory_binding_ambiguous:{name}")
                override = self.memory_bindings[matches[0][1]]
        if override is not None:
            result.update({str(key): value
                           for key, value in dict(override).items()
                           if value is not None})
        return result or None

    def _submit_task(
        self, name: str, work: Mapping[str, float], after: Sequence[Event],
        acquire: Sequence[Allocation], free: Sequence[Allocation],
        priority: int, coupled: bool, on_start, on_complete, owner: str,
        memory_bindings: Mapping[str, Mapping] | None = None,
    ) -> Event:
        metadata = {"memory_replica": self.memory_replica,
                    "memory_stage": self.memory_stage,
                    "memory_owner": self.memory_owner,
                    "memory_storage": self.memory_storage,
                    "memory_ranges": self.memory_ranges,
                    "memory_generation": self.memory_generation,
                    "memory_event": self.memory_event,
                    "memory_host_bytes": self.memory_host_bytes,
                    "memory_unit": self.memory_unit}
        predecessors = tuple(after)
        wrapped_start = None
        if on_start is not None:
            def wrapped_start(event: Event) -> None:
                self.state._run_read_scope(event.sequence, on_start, event)
        wrapped_complete = None
        if on_complete is not None:
            def wrapped_complete(event: Event) -> None:
                at = event.time if event.time is not None else self.engine.now
                ancestors = self.state._collect_causal_ancestors(predecessors, at=at)
                emitted = self.state._run_causal_scope(
                    ancestors, on_complete, event, at=at,
                )
                self.state._record_event_commits(
                    event.sequence, ancestors | emitted, at=at,
                )
        return self.engine.submit(
            name, work, after=predecessors, acquire=tuple(acquire),
            free=tuple(free), priority=priority, coupled=coupled,
            on_start=wrapped_start, on_complete=wrapped_complete, owner=owner,
            metadata={key: value for key, value in metadata.items()
                      if value is not None},
            memory=self.memory, memory_pools=self.memory_pools,
            memory_bindings=memory_bindings)

    def _preflight_event_publication(
        self, fragment: ProgramFragment,
        event_bindings: Mapping[str, Any],
    ) -> dict[str, int]:
        """Validate the whole event_out batch without touching the registry.

        The union of existing names and every name this fragment will publish
        must fit ``event_limit``. Every proposed generation must be unused, and
        every explicit binding must resolve. A rejection therefore happens
        before an Engine task, a live handle, an eviction or a sequence bump;
        exactly-fitting batches still publish normally.
        """
        ports = fragment.ports.event_out
        output_names = {port.name for port in ports}
        unknown = set(event_bindings) - output_names
        if unknown:
            raise FragmentContractError(
                f"fragment_event_binding_unknown:{sorted(unknown)[0]}")
        existing_names = set(self._generations)
        if len(existing_names | output_names) > self.event_limit:
            raise FragmentContractError(
                f"fragment_event_name_limit_exceeded:{self.event_limit}")
        existing_keys = set(self._events)
        proposed_keys: set[tuple[str, int]] = set()
        result: dict[str, int] = {}
        for port in ports:
            generation = port.generation
            if generation is None:
                generation = self._generations.get(port.name, -1) + 1
            key = port.name, generation
            if key in existing_keys or key in proposed_keys:
                raise FragmentContractError(
                    f"fragment_event_generation_already_recorded:"
                    f"{port.name}:{generation}")
            if port.name in event_bindings:
                self._as_event(event_bindings[port.name],
                               f"event:{port.name}")
            result[port.name] = generation
            proposed_keys.add(key)
        return result

    def _unique_allocation_event(self, kind: str, port_name: str,
                                 table: Mapping[str, int]) -> tuple[str, str]:
        if not table:
            raise UnsupportedFragmentSemantics(
                f"fragment_event_port_unbound:{kind}:{port_name}")
        if port_name in table:
            if table[port_name] != 1:
                raise UnsupportedFragmentSemantics(
                    f"fragment_event_port_ambiguous:{kind}:{port_name}")
            return kind, port_name
        if len(table) == 1:
            name, count = next(iter(table.items()))
            if count == 1:
                return kind, name
        raise UnsupportedFragmentSemantics(
            f"fragment_event_port_ambiguous:{kind}:{port_name}")

    def _plan_event_outputs(
        self, fragment: ProgramFragment, plan: _AllocationPlan,
        conditions: Mapping[str, bool],
        event_bindings: Mapping[str, Any],
    ) -> dict[str, tuple[str, ...]]:
        output_names = {port.name for port in fragment.ports.event_out}
        unknown = set(event_bindings) - output_names
        if unknown:
            raise FragmentContractError(
                f"fragment_event_binding_unknown:{sorted(unknown)[0]}")
        grants: dict[str, int] = {}
        releases: dict[str, int] = {}
        for planned in plan.program:
            grants[planned.name] = 1
            releases[planned.name] = 1
        for planned in plan.stage:
            grants[planned.name] = fragment.repeats
            if planned.external_hold is not None:
                releases[planned.name] = 1
            elif planned.local_release:
                releases[planned.name] = fragment.repeats
        result: dict[str, tuple[str, ...]] = {}
        for port in fragment.ports.event_out:
            if port.name in event_bindings:
                self._as_event(event_bindings[port.name],
                               f"event:{port.name}")
                result[port.name] = ("explicit", port.name)
                continue
            kind = port.event
            if kind in {"complete", "ready"}:
                result[port.name] = ("work",)
            elif kind == "grant":
                result[port.name] = self._unique_allocation_event(
                    kind, port.name, grants)
            elif kind == "release":
                result[port.name] = self._unique_allocation_event(
                    kind, port.name, releases)
            elif kind == "commit":
                if not any(
                        effect.action == "commit"
                        and self._active(effect.condition, conditions,
                                         effect.target)
                        for effect in fragment.state_effects):
                    raise UnsupportedFragmentSemantics(
                        f"fragment_event_port_without_commit:{port.name}")
                result[port.name] = ("work",)
            else:
                raise UnsupportedFragmentSemantics(
                    f"fragment_event_port_requires_explicit_binding:"
                    f"{kind}:{port.name}")
        return result

    def _resolve_output_events(
        self, fragment: ProgramFragment, plan: Mapping[str, tuple[str, ...]],
        work_complete: Event, grant_by_name: Mapping[str, Event],
        release_by_name: Mapping[str, Event],
        event_bindings: Mapping[str, Any],
    ) -> dict[str, Event]:
        result: dict[str, Event] = {}
        for port in fragment.ports.event_out:
            key = plan[port.name]
            if key[0] == "explicit":
                result[port.name] = self._as_event(
                    event_bindings[port.name], f"event:{port.name}")
            elif key[0] == "work":
                result[port.name] = work_complete
            elif key[0] == "grant":
                event = grant_by_name.get(key[1])
                if event is None:
                    raise FragmentContractError(
                        f"fragment_event_grant_missing:{key[1]}")
                result[port.name] = event
            elif key[0] == "release":
                event = release_by_name.get(key[1])
                if event is None:
                    raise FragmentContractError(
                        f"fragment_event_release_missing:{key[1]}")
                result[port.name] = event
        return result

    def _record_outputs(self, fragment: ProgramFragment,
                        events: Mapping[str, Event],
                        generations: Mapping[str, int]) -> None:
        for port in fragment.ports.event_out:
            self.record_event(port.name, events[port.name],
                              generations[port.name])

    def _record_unsupported(self, reason: str) -> None:
        if reason in self._unsupported:
            self._unsupported[reason] += 1
            return
        if len(self._unsupported) >= 32:
            self._unsupported_overflow += 1
            return
        self._unsupported[reason] = 1

    def _finish(self, record: _LiveRecord, event: Event) -> None:
        handle = record.handle
        completion_event = handle.event if handle is not None else event
        at = (
            completion_event.time
            if completion_event.time is not None
            else self.engine.now
        )
        linked = [completion_event.sequence]
        if handle is not None:
            linked.extend(rel.sequence for rel in handle.release_events)
        self.state._link_active_event_commits(linked, at=at)
        self._live.pop(completion_event.sequence, None)
        fragment = record.fragment
        release_boundaries = len(handle.release_by_name) if handle is not None else 0
        grants = tuple(handle.grant_by_name) if handle is not None else ()
        service_work = self._work(fragment, fragment.repeats)
        useful_work = self._work(fragment, fragment.repeats, kind="useful")
        start_ms = record.started_ms if record.started_ms is not None else event.started
        observation = {
            "name": fragment.name,
            "sequence": record.sequence,
            "completion_ms": completion_event.time,
            "start_ms": start_ms,
            "repeats": fragment.repeats,
            "batched": bool(handle and handle.batched),
            "service_work": dict(service_work),
            "useful_work": dict(useful_work),
            "grant_events": len(grants),
            "release_events": release_boundaries,
            "state": self.state.snapshot(),
        }
        self._totals["fragments"] += 1
        self._totals["repeats"] += fragment.repeats
        if handle and handle.batched:
            self._totals["batched"] += 1
        else:
            self._totals["expanded"] += 1
        self._totals["release_events"] += release_boundaries
        for resource, amount in service_work.items():
            self._total_service[resource] = (
                self._total_service.get(resource, 0.0) + float(amount))
        if self._recent is not None:
            if len(self._recent) >= self.cache_limit:
                self._recent.popleft()
                self._recent_dropped += 1
            self._recent.append(observation)
        aggregate = self._aggregates.pop(fragment.name, None)
        if aggregate is None:
            aggregate = {
                "count": 0, "repeats": 0, "batched": 0,
                "first_ms": start_ms, "last_ms": completion_event.time,
                "min_span_ms": None, "max_span_ms": None,
            }
        aggregate["count"] += 1
        aggregate["repeats"] += fragment.repeats
        aggregate["batched"] += int(bool(handle and handle.batched))
        aggregate["last_ms"] = completion_event.time
        if start_ms is not None and completion_event.time is not None:
            span = completion_event.time - start_ms
            previous = aggregate["min_span_ms"]
            aggregate["min_span_ms"] = span if previous is None else min(previous, span)
            previous = aggregate["max_span_ms"]
            aggregate["max_span_ms"] = span if previous is None else max(previous, span)
        self._aggregates[fragment.name] = aggregate
        while len(self._aggregates) > self.cache_limit:
            self._aggregates.popitem(last=False)
            self._aggregates_evicted += 1
