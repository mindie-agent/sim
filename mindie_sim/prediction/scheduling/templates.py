"""Finite ordered and indexed pipeline execution.

[Role] Submit declared setup, tile stages and drain on the caller's Engine.
[Invariants] Allocation scopes, dependencies and stage work are preserved for each tile.
[Timeouts & Locks] Only a slot window is in flight. Indexed phases are resolved on
  demand.
[Side-Effect Policy] Submits Engine tasks and maintains periodic bookkeeping.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping as _AbcMapping
from dataclasses import replace
from typing import Mapping, Sequence

from ..engine import Allocation, Engine, Event
from .state import FragmentContractError


def _scoped_allocations(
    label: str, items: Sequence[Allocation], *,
    root: str | None = None, bindings: Mapping[str, str] | None = None,
    persistent: frozenset[str] = frozenset(),
    slot: int | None = None,
) -> tuple[Allocation, ...]:
    """Bind allocations by the template's invocation/output/panel scope.

    ``bindings`` comes from ``PipelineTemplate.allocation_bindings``. The
    invocation scope is the explicit template root (never derived from a display
    task name); output allocations span panels of one output; panel allocations
    remain tile-local. Setup/body/drain use this same function.
    """
    scope_root = root or _pipeline_scope(label)
    table = dict(bindings or {})
    result = []
    for item in items:
        scope = table.get(item.name)
        if scope == "invocation":
            prefix = scope_root
        elif scope == "output":
            digits = "".join(ch for ch in item.name if ch.isdigit())
            prefix = f"{scope_root}:output{digits or 0}"
        elif scope == "slot" and slot is not None:
            prefix = f"{scope_root}:slot{slot}"
        elif scope == "panel":
            prefix = label
        elif item.name in persistent:
            prefix = scope_root
        else:
            prefix = label
        result.append(Allocation(f"{prefix}:{item.name}", item.pool,
                                 item.size, item.units))
    return tuple(result)


def _indexed_scoped_allocations(
    items: Sequence[Allocation], *, launch_uid: str, owner: int, index: int,
    output: int | None, panel: int | None, slot: int | None,
    bindings: Mapping[str, Mapping],
    persistent: frozenset[str],
) -> tuple[Allocation, ...]:
    """Bind one indexed tile's allocations by real scope ordinal.

    The producer's binding row is authoritative.  A row with an explicit
    physical identity is used verbatim; otherwise the local key is
    ``launch_uid/owner/scope/ordinal/symbol``.  Indexed symbols are stable
    names such as ``panel``/``accumulator``, so no digit parsing is possible.
    """
    result = []
    for item in items:
        row = bindings.get(item.name)
        row = row if isinstance(row, _AbcMapping) else {}
        scope = row.get("scope")
        explicit = None
        for key in ("identity", "physical", "physical_name", "external"):
            value = row.get(key)
            if isinstance(value, str) and value:
                explicit = value
                break
        if explicit is not None:
            name = explicit
        elif scope == "invocation":
            name = f"{launch_uid}:owner{owner}:invocation0:{item.name}"
        elif scope == "output":
            ordinal = output if output is not None else int(
                row.get("ordinal", 0))
            name = f"{launch_uid}:owner{owner}:output{ordinal}:{item.name}"
        elif scope == "slot":
            ordinal = slot if slot is not None else int(row.get("ordinal", 0))
            name = f"{launch_uid}:owner{owner}:slot{ordinal}:{item.name}"
        elif scope == "panel" or item.name not in persistent:
            # A panel is one concrete absolute panel, not a recurring wave
            # residue: using the absolute index keeps overlapping slots apart.
            ordinal = index if index is not None else int(row.get("ordinal", 0))
            name = f"{launch_uid}:owner{owner}:panel{ordinal}:{item.name}"
        else:
            name = f"{launch_uid}:owner{owner}:invocation0:{item.name}"
        result.append(Allocation(name, item.pool, item.size, item.units))
    return tuple(result)


def _pipeline_scope(label: str) -> str:
    # ``label`` is a tile task name (name:tileN) or a setup/drain label.
    marker = ":tile"
    index = label.find(marker)
    return label[:index] if index >= 0 else label.split(":setup", 1)[0].split(":drain", 1)[0]


def _drop_periodic_run(engine: Engine, run_id: int) -> None:
    engine._periodic_runs.pop(run_id, None)
    for sequence, metadata in tuple(engine._periodic_tasks.items()):
        if metadata and metadata[0] == run_id:
            engine._periodic_tasks.pop(sequence, None)


def _reachable_phase_stage_groups(
    phase, *, indexed: bool, cursor: int
) -> tuple[tuple, ...]:
    """Return every stage group actually reachable during ``phase``."""
    residue = tuple(getattr(phase, "residue_stages", ()) or ())
    if not residue:
        return (tuple(getattr(phase, "stages", ())),)
    if indexed:
        return tuple(tuple(group) for group in residue)
    count = int(getattr(phase, "count", 0))
    period = len(residue)
    selector = getattr(phase, "stages_for", None)
    if callable(selector):
        return tuple(
            tuple(selector(cursor + offset))
            for offset in range(min(count, period))
        )
    return tuple(
        tuple(residue[(cursor + offset) % period])
        for offset in range(min(count, period))
    )


def _uniform_template_stages(phases) -> tuple | None:
    """Return the single stage tuple if every reachable tile across ``phases`` is identical."""
    batch_stages = None
    cursor = 0
    for phase in phases:
        count = int(getattr(phase, "count", 0))
        for group in _reachable_phase_stage_groups(phase, indexed=False, cursor=cursor):
            if batch_stages is None:
                batch_stages = group
            elif group != batch_stages:
                return None
        cursor += count
    return batch_stages


class _PipelineTemplateRun:
    """Retain the bounded live state shared by a template's Engine callbacks."""

    def __init__(self, engine, template, name, owner, phases, slots, indexed, index_of, total):
        self.engine = engine
        self.template = template
        self.name = name
        self.owner = owner
        self.phases = phases
        self.slots = slots
        self.indexed = indexed
        self.index_of = index_of
        self.total = total
        # Names acquired and freed at different phase or residue boundaries are
        # resident across tiles (accumulator/KV/workspace). Names acquired and
        # freed inside the same reachable stage group are tile-local.
        acquire_group: dict[str, set[tuple[int, int]]] = {}
        free_group: dict[str, set[tuple[int, int]]] = {}
        has_previous_lags = False
        max_lag = 1
        cursor = 0
        for phase_index, phase in enumerate(phases):
            groups = _reachable_phase_stage_groups(
                phase, indexed=indexed, cursor=cursor)
            cursor += int(getattr(phase, "count", 0))
            for group_index, group in enumerate(groups):
                group_id = (phase_index, group_index)
                for stage in group:
                    for allocation in tuple(getattr(stage, "acquire", ())):
                        acquire_group.setdefault(allocation.name, set()).add(group_id)
                    for allocation in tuple(getattr(stage, "free", ())):
                        free_group.setdefault(allocation.name, set()).add(group_id)
                    for _, lag in getattr(stage, "previous_lags", ()):
                        has_previous_lags = True
                        if lag > max_lag:
                            max_lag = lag
        persistent_names = frozenset(
            name for name, acquired in acquire_group.items()
            if free_group.get(name) != acquired)
        binding_rows = tuple(getattr(template, "allocation_bindings", ()) or ())
        allocation_bindings = {
            str(row.get("name")): str(row.get("scope"))
            for row in binding_rows if isinstance(row, _AbcMapping) and row.get("name")}
        indexed_binding_rows = {
            str(row.get("name")): dict(row)
            for row in binding_rows
            if isinstance(row, _AbcMapping) and row.get("name")}
        # Immutable, allocation-free Stage validation can be shared by every
        # repeat. Acquire/free stages still bind each occurrence's allocation id and
        # therefore keep Stage.prepare dynamic.
        prepared_cache: dict[tuple, object] = {}
        setup = tuple(getattr(template, "setup", ()))
        drain = tuple(getattr(template, "drain", ()))
        for stage in drain:
            for _, lag in getattr(stage, "previous_lags", ()):
                has_previous_lags = True
                if lag > max_lag:
                    max_lag = lag
        # Setup/body/drain share the same invocation symbols. The producer's
        # allocation_bindings scans body phases; any allocation first visible in
        # setup/drain is invocation-scoped here rather than getting a new label.
        template_owner = getattr(template, "owner", 0)
        if indexed and (type(template_owner) is not int or template_owner < 0):
            raise FragmentContractError("pipeline_template_owner_invalid")
        for stage in (*setup, *drain):
            for allocation in (*stage.acquire, *stage.free):
                allocation_bindings.setdefault(allocation.name, "invocation")
                indexed_binding_rows.setdefault(allocation.name, {
                    "name": allocation.name, "scope": "invocation",
                    "owner": template_owner,
                })
        self.persistent_names = persistent_names
        self.allocation_bindings = allocation_bindings
        self.indexed_binding_rows = indexed_binding_rows
        self.prepared_cache = prepared_cache
        self.setup = setup
        self.drain = drain
        self.template_owner = template_owner
        self.has_previous_lags = has_previous_lags
        self.max_lag = max_lag
        self._track_periodic = not has_previous_lags
        self._tile_events: dict[int, dict[str, Event]] = {}
        self._uses_slot_scope = (not indexed) and any(
            scope == "slot" for scope in allocation_bindings.values())
        self._reusable_alloc_names = frozenset(
            alloc_name for alloc_name, row in indexed_binding_rows.items()
            if row.get("scope") == "slot"
            or any(
                isinstance(row.get(key), str) and row.get(key)
                for key in ("identity", "physical", "physical_name", "external")
            )
        ) if indexed else frozenset()
        self._slot_alloc_free: dict[str, Event] = {}
        self._free_slots: deque[int] = deque(range(slots))
        self._retire_slot: dict[int, int] = {}
        self._last_slot: int = 0

    def _prepared_for(self, stage):
        engine = self.engine
        prepared_cache = self.prepared_cache
        if stage.acquire or stage.free or stage.program is not None:
            return None
        cohort_members = getattr(stage, "cohort_members", ())
        members_key = (
            tuple(sorted((res, tuple(tuple(row) for row in rows))
                         for res, rows in cohort_members.items()))
            if isinstance(cohort_members, _AbcMapping)
            else tuple(cohort_members)
        )
        key = (
            stage.name,
            tuple(stage.work.items()),
            stage.latency,
            tuple(sorted(stage.caps.items())),
            tuple(sorted((res, tuple(units))
                         for res, units in stage.placement.items())),
            stage.coupled,
            stage.priority,
            tuple(sorted(stage.work_units.items())),
            bool(getattr(stage, "latency_from_first_service", False)),
            tuple(getattr(stage, "deliveries", ())),
            int(getattr(stage, "cohort_size", 1)),
            tuple(getattr(stage, "cohort_widened", ())),
            members_key,
            bool(getattr(stage, "keep_empty", False)),
            tuple(sorted(getattr(stage, "metadata", {}).items())),
        )
        prepared = prepared_cache.get(key)
        if prepared is None:
            prepared = stage.prepare(engine, stage.name, lambda items: ())
            prepared_cache[key] = prepared
        return prepared

    def _scoped_at(self, label: str, index: int, output: int | None,
                   panel: int | None, slot: int | None):
        name, persistent_names = self.name, self.persistent_names
        if self.indexed:
            return lambda items: _indexed_scoped_allocations(
                items, launch_uid=name, owner=self.template_owner, index=index,
                output=output, panel=panel, slot=slot,
                bindings=self.indexed_binding_rows, persistent=persistent_names)
        resolved_slot = slot if slot is not None else index % self.slots
        return lambda items: _scoped_allocations(
            label, items, root=name, bindings=self.allocation_bindings,
            persistent=persistent_names, slot=resolved_slot)

    def _indexed_mapping(self, index: int):
        mapped = self.index_of(index)
        if not isinstance(mapped, tuple) or len(mapped) != 5:
            raise FragmentContractError("pipeline_index_mapping_invalid")
        phase_id, end, output, panel, slot = mapped
        if (type(phase_id) is not int or not 0 <= phase_id < len(self.phases)
                or type(end) is not int or not index < end <= self.total
                or type(output) is not int or output < 0
                or type(panel) is not int or panel < 0
                or type(slot) is not int or not 0 <= slot < self.slots):
            raise FragmentContractError("pipeline_index_mapping_invalid")
        return phase_id, end, output, panel, slot

    def start(self, ready, quiescent):
        engine, template, name, owner = self.engine, self.template, self.name, self.owner
        phases, indexed, total = self.phases, self.indexed, self.total
        setup, persistent_names = self.setup, self.persistent_names
        setup_ms = float(getattr(template, "setup_ms", 0.0))
        minimum_ms = float(getattr(template, "minimum_ms", 0.0))
        if setup_ms < 0 or minimum_ms < 0:
            raise FragmentContractError("pipeline_template_negative_fixed_time")
        drain_index = total - 1 if total else 0
        if indexed and total:
            (_drain_phase, _drain_end, drain_output,
             drain_panel, drain_slot) = self._indexed_mapping(drain_index)
        else:
            drain_output = drain_panel = drain_slot = 0

        self.drain_coordinates = (drain_index, drain_output, drain_panel, drain_slot)
        current = ready
        if setup_ms:
            current = engine.submit(name + ":setup", after=(ready,),
                                    latency=setup_ms, owner=owner)
        if setup:
            local: dict[str, Event] = {}
            for stage in setup:
                deps = tuple(local[key] for key in stage.after)
                scoped = self._scoped_at(name + ":setup:" + stage.name, 0, 0, 0, 0)
                local[stage.name] = stage.submit(
                    engine, f"{name}:setup:{stage.name}", deps or (current,),
                    scoped, owner, prepared=self._prepared_for(stage))
            current = engine.submit(name + ":setup_complete",
                                    after=tuple(local.values()) or (current,),
                                    owner=owner)
        observation = (engine.submit(name + ":span", after=(current,),
                                     latency=minimum_ms, owner=owner)
                       if minimum_ms > 0 else current)
        complete = engine.future(name + ":complete")
        run_id = complete.sequence
        periodic_run = {
            "run_id": run_id, "template": template, "name": name, "owner": owner,
            "phase_ordinal": 0,
            "phase_end": int(getattr(phases[0], "count", 0)) if phases else 0,
            "next": 0, "finished": 0, "slots": (),
            "live_tiles": set(), "retire_tiles": {}, "stage_tail": {}, "retire": None,
            "indexed": indexed,
        }
        if self._track_periodic:
            engine._periodic_runs[run_id] = periodic_run
        self.current = current
        self.complete = complete
        self.periodic_run = periodic_run
        self.run_id = run_id
        if not self._try_batch(quiescent):
            periodic_run.update({
                "next": 0, "finished": 0, "phase_ordinal": 0,
                "phase_end": int(getattr(phases[0], "count", 0)) if phases else 0,
                "stage_tail": {}, "last_events": (current,), "retire": None,
                "total": total, "persistent_names": persistent_names,
            })
            engine.submit(name + ":begin", after=(current,), on_complete=self._begin,
                          owner=owner)
        return engine.submit(name + ":complete_outer",
                             after=(complete, observation), owner=owner)

    def _try_batch(self, quiescent):
        engine, name, owner = self.engine, self.name, self.owner
        phases, indexed, total = self.phases, self.indexed, self.total
        current, periodic_run, run_id = self.current, self.periodic_run, self.run_id
        allocation_bindings, persistent_names = self.allocation_bindings, self.persistent_names
        if not (quiescent and not indexed and phases and total > 1):
            return False
        batch_stages = _uniform_template_stages(phases)
        if not batch_stages or len(batch_stages) != 1:
            return False
        stage = batch_stages[0]
        # A fully quiescent, dependency-free, allocation-free homogeneous
        # single-stage run can be integrated as one meta-cycle only when all
        # active resources progress together (single non-aliased resource or
        # coupled multi-resource) so intermediate horizons and deliveries match
        # tile-by-tile execution exactly.
        batch_count = total
        active = [res for res, amount in stage.work.items() if amount]
        aliases = getattr(engine, "_hbm_bus_aliases", None) or {}
        cohort_size = int(getattr(stage, "cohort_size", 1))
        batchable = (
            not stage.after
            and not stage.previous
            and not getattr(stage, "previous_lags", ())
            and not stage.acquire
            and not stage.free
            and stage.program is None
            and stage.latency == 0
            and (len(active) <= 1 or stage.coupled)
            and (stage.coupled or not any(res in aliases for res in active))
            and (cohort_size == 1 or all(
                engine.resources[res].policy == "fair"
                for res in active if res in engine.resources))
        )
        if not batchable:
            return False
        barrier = []
        for stage in batch_stages:
            work = {resource: amount * batch_count
                    for resource, amount in stage.work.items()}
            raw_deliveries = getattr(stage, "deliveries", ())
            if raw_deliveries:
                deliveries = tuple(
                    (counter, amount * batch_count)
                    for counter, amount in raw_deliveries
                )
                batched = replace(stage, work=work, deliveries=deliveries)
            else:
                batched = replace(stage, work=work)
            barrier.append(batched.submit(
                engine, f"{name}:batch:{stage.name}",
                (current,), lambda items: _scoped_allocations(
                    name, items, root=name,
                    bindings=allocation_bindings), owner,
                prepared=self._prepared_for(batched)))
        periodic_run.update({
            "next": total, "finished": total, "phase_ordinal": 0,
            "phase_end": total, "stage_tail": {},
            "last_events": tuple(barrier) or (current,),
            "total": total, "persistent_names": persistent_names,
            "batch_mode": True,
        })
        batch_complete = engine.submit(
            name + ":batch_complete", after=tuple(barrier) or (current,),
            on_complete=self._drain_and_resolve, owner=owner)
        if self._track_periodic:
            engine._periodic_tasks[batch_complete.sequence] = (
                run_id, total, len(batch_stages), "batch")
        periodic_run["slots"] = ()
        return True

    def _select_stages(self, index: int):
        indexed, phases, periodic_run = self.indexed, self.phases, self.periodic_run
        if indexed:
            phase_id, end, output, panel, slot = self._indexed_mapping(index)
            # The producer's end is an absolute exclusive phase bound. Keep it
            # for observation, but choose the class afresh for every absolute
            # index: class ids are catalog ids, not an ordered cursor.
            periodic_run["phase_ordinal"] = phase_id
            periodic_run["phase_end"] = end
            phase = phases[phase_id]
            selector = getattr(phase, "stages_for", None)
            stages = (selector(index) if callable(selector)
                      else getattr(phase, "stages"))
            selected = tuple(stages)
            if not selected:
                raise FragmentContractError(
                    "pipeline_indexed_phase_requires_stages")
            return selected, (output, panel, slot), 1
        while (periodic_run["phase_ordinal"] + 1 < len(phases)
               and index >= periodic_run["phase_end"]):
            ordinal = periodic_run["phase_ordinal"] + 1
            periodic_run["phase_ordinal"] = ordinal
            periodic_run["phase_end"] += int(
                getattr(phases[ordinal], "count", 0))
        phase = phases[periodic_run["phase_ordinal"]]
        selector = getattr(phase, "stages_for", None)
        stages = (selector(index) if callable(selector)
                  else getattr(phase, "stages"))
        selected = tuple(stages)
        if not selected:
            raise FragmentContractError("pipeline_template_phase_requires_stages")
        residue = tuple(getattr(phase, "residue_stages", ()) or ())
        period = len(residue) if residue else 1
        return selected, None, period

    def _reusable_acquire_deps(self, scoped, stage) -> list[Event]:
        if not self._reusable_alloc_names or not stage.acquire:
            return []
        result: list[Event] = []
        for raw, bound in zip(stage.acquire, scoped(stage.acquire)):
            if raw.name in self._reusable_alloc_names:
                prior = self._slot_alloc_free.get(bound.name)
                if prior is not None:
                    result.append(prior)
        return result

    def _record_reusable_frees(self, scoped, stage, event: Event) -> None:
        if not self._reusable_alloc_names or not stage.free:
            return
        for raw, bound in zip(stage.free, scoped(stage.free)):
            if raw.name in self._reusable_alloc_names:
                self._slot_alloc_free[bound.name] = event

    def _finish(self, _event: Event) -> None:
        self._tile_events.clear()
        self._retire_slot.clear()
        self._slot_alloc_free.clear()
        self.engine.resolve(self.complete)
        if self._track_periodic:
            _drop_periodic_run(self.engine, self.run_id)

    def _drain_and_resolve(self, _event=None) -> None:
        if not self.drain:
            self._finish(_event)
            return
        engine, name, owner = self.engine, self.name, self.owner
        periodic_run = self.periodic_run
        drain_index, drain_output, drain_panel, drain_slot = self.drain_coordinates
        if not self.indexed:
            drain_slot = self._last_slot
        local: dict[str, Event] = {}
        for stage in self.drain:
            deps = [local[key] for key in stage.after]
            deps.extend(periodic_run["stage_tail"][key]
                        for key in stage.previous
                        if key in periodic_run["stage_tail"])
            for dep_name, lag in getattr(stage, "previous_lags", ()):
                target = self.total - lag
                if target >= 0:
                    prior_tile = self._tile_events.get(target)
                    if prior_tile is not None and dep_name in prior_tile:
                        deps.append(prior_tile[dep_name])
            scoped = self._scoped_at(name + ":drain:" + stage.name, drain_index,
                               drain_output, drain_panel, drain_slot)
            if self.indexed:
                deps.extend(self._reusable_acquire_deps(scoped, stage))
            event = stage.submit(
                engine, f"{name}:drain:{stage.name}",
                tuple(dict.fromkeys(deps)) or periodic_run["last_events"], scoped, owner,
                prepared=self._prepared_for(stage))
            local[stage.name] = event
            if self.indexed:
                self._record_reusable_frees(scoped, stage, event)
        engine.submit(name + ":drain_complete", after=tuple(local.values()),
                      on_complete=self._finish, owner=owner)

    def _retire(self, event: Event) -> None:
        periodic_run = self.periodic_run
        finished = periodic_run["finished"] + 1
        periodic_run["finished"] = finished
        tile_index = periodic_run["retire_tiles"].pop(event.sequence, None)
        if tile_index is not None:
            periodic_run["live_tiles"].discard(tile_index)
        freed_slot = self._retire_slot.pop(event.sequence, None)
        if freed_slot is not None:
            self._free_slots.append(freed_slot)
        periodic_run["slots"] = tuple(sorted(
            tile - finished for tile in periodic_run["live_tiles"]))
        if finished == self.total:
            self._drain_and_resolve()
            return
        self._issue()

    def _issue(self) -> None:
        periodic_run = self.periodic_run
        index = periodic_run["next"]
        if index >= self.total:
            return
        stages, coordinates, period = self._select_stages(index)
        engine, name, owner = self.engine, self.name, self.owner
        indexed, current, run_id = self.indexed, self.current, self.run_id
        periodic_run["next"] = index + 1
        local: dict[str, Event] = {}
        # One immutable snapshot per panel: a same-panel ``after`` edge never
        # masquerades as the previous panel's stream tail.
        previous = dict(periodic_run["stage_tail"])
        label = f"{name}:tile{index}"
        assigned_slot: int | None = None
        if indexed:
            output, panel, slot = coordinates
            scoped = self._scoped_at(label, index, output, panel, slot)
        else:
            slot = None
            assigned_slot = self._free_slots.popleft()
            self._last_slot = assigned_slot
            scoped = self._scoped_at(label, index, None, None, assigned_slot)
        suffix = ""
        if period > 1:
            suffix += f":r{index % period}"
        if self._uses_slot_scope:
            suffix += f":s{assigned_slot}"
        elif indexed and self._reusable_alloc_names:
            suffix += f":s{slot}"
        stage_kind = f"stage{suffix}"
        retire_kind = f"retire{suffix}"
        for stage in stages:
            deps = [local[key] for key in stage.after]
            deps.extend(previous[key] for key in stage.previous
                        if key in previous)
            for dep_name, lag in getattr(stage, "previous_lags", ()):
                target = index - lag
                if target >= 0:
                    prior_tile = self._tile_events.get(target)
                    if prior_tile is not None and dep_name in prior_tile:
                        deps.append(prior_tile[dep_name])
            if stage.name in previous:
                deps.append(previous[stage.name])
            if indexed:
                deps.extend(self._reusable_acquire_deps(scoped, stage))
            event = stage.submit(
                engine, f"{label}:{stage.name}",
                tuple(dict.fromkeys(deps)) or (current,), scoped, owner,
                prepared=self._prepared_for(stage))
            local[stage.name] = event
            periodic_run["stage_tail"][stage.name] = event
            if indexed:
                self._record_reusable_frees(scoped, stage, event)
            if self._track_periodic:
                engine._periodic_tasks[event.sequence] = (
                    run_id, index, len(local) - 1, stage_kind)
        periodic_run["last_events"] = tuple(local.values())
        retire_event = engine.submit(f"{label}:retire",
                                     after=periodic_run["last_events"],
                                     on_complete=self._retire, owner=owner)
        periodic_run["retire"] = retire_event
        if assigned_slot is not None:
            self._retire_slot[retire_event.sequence] = assigned_slot
        if self._track_periodic:
            engine._periodic_tasks[retire_event.sequence] = (
                run_id, index, len(stages), retire_kind)
        periodic_run["retire_tiles"][retire_event.sequence] = index
        periodic_run["live_tiles"].add(index)
        if self.has_previous_lags:
            self._tile_events[index] = local
            cutoff = index - max(self.slots, self.max_lag)
            for old_idx in tuple(self._tile_events):
                if old_idx < cutoff and old_idx not in periodic_run["live_tiles"]:
                    del self._tile_events[old_idx]
        periodic_run["slots"] = tuple(sorted(
            tile - periodic_run["finished"]
            for tile in periodic_run["live_tiles"]))

    def _begin(self, _event: Event) -> None:
        if not self.total:
            self._drain_and_resolve()
            return
        for _ in range(min(self.slots, self.total)):
            self._issue()


def start_pipeline_template(
    engine: Engine, template, name: str, ready: Event, *, owner: str = "",
    quiescent: bool = False,
) -> Event:
    """Submit a finite ``PipelineTemplate`` lazily on the caller's Engine.

    This is the T04 side of the T10 handoff. It consumes the duck-typed
    ``PipelinePhase(stages, count)`` / ``PipelineTemplate(slots, phases, setup,
    drain, setup_ms, minimum_ms)`` contract without importing ``program.py``.
    At most ``slots`` tiles are in flight; each retire issues the next real tile.
    ``Stage.after``/``previous``, per-stage work/caps/placement/priority,
    coupled progress and per-tile scoped allocations are preserved. No phase is
    averaged into one body and no whole-phase allocation maximum is reserved.
    An indexed template is selected by absolute ``index_of(i)`` and
    ``phases[phase_id].stages_for(i)``; it never materializes every panel.
    """
    if not isinstance(engine, Engine):
        raise FragmentContractError("pipeline_template_requires_engine")
    phases = tuple(getattr(template, "phases", ()))
    slots = int(getattr(template, "slots", 0))
    if slots < 1:
        raise FragmentContractError("pipeline_template_requires_positive_slots")
    indexed_panels = getattr(template, "indexed_panels", None)
    index_of = getattr(template, "index_of", None)
    has_indexed_panels = indexed_panels is not None
    has_index_of = index_of is not None
    if has_indexed_panels != has_index_of:
        raise FragmentContractError(
            "pipeline_indexed_template_requires_both_fields")
    indexed = has_indexed_panels and has_index_of
    if indexed:
        if (type(indexed_panels) is not int or indexed_panels < 0
                or not callable(index_of)):
            raise FragmentContractError("pipeline_indexed_panels_invalid")
        if any(type(getattr(phase, "count", 0)) is not int
               or getattr(phase, "count") != 1 for phase in phases):
            raise FragmentContractError(
                "pipeline_indexed_phase_count_must_be_one")
        if any(getattr(phase, "end", None) is not None for phase in phases):
            raise FragmentContractError(
                "pipeline_indexed_phase_end_must_be_none")
        if indexed_panels and not phases:
            raise FragmentContractError(
                "pipeline_indexed_template_requires_catalog")
        total = indexed_panels
    else:
        total = sum(int(getattr(phase, "count", 0)) for phase in phases)
        if total < 0:
            raise FragmentContractError("pipeline_template_count_invalid")
    for phase in phases:
        residue = tuple(getattr(phase, "residue_stages", ()) or ())
        if (not tuple(getattr(phase, "stages", ()))
                or any(not tuple(group) for group in residue)):
            raise FragmentContractError("pipeline_template_phase_requires_stages")
        if int(getattr(phase, "count", 0)) < 1:
            raise FragmentContractError("pipeline_template_phase_requires_positive_count")
    run = _PipelineTemplateRun(engine, template, name, owner, phases, slots, indexed, index_of, total)
    return run.start(ready, quiescent)


# Capability evidence is the real callable's integer marker, never a producer
# self-report.  The indexed reader above is connected before this is published;
# old readers still see no marker and keep the exact legacy encoding.
start_pipeline_template.indexed_panels_abi = 1
