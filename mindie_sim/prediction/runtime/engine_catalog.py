"""Unique ProgramCatalog preflight and bounded-window submission for Engine.

[Role] Validate and materialize flat and finite repeated ``ProgramCatalog``
  specifications onto the single causal Engine without unbounded Event creation.
[Invariants] Repeated catalogs materialize at most ``finite.slots`` live panels;
  preflight is independent of ``finite.total`` above ``_FULL_FINITE_PREFLIGHT``.
[Timeouts & Locks] Bounded preflight and live panel window; no locks or I/O.
[Side-Effect Policy] Submits tasks, promises and state-ledger commits on Engine.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

from .contracts import (
    AllocationKey, BindingFrame, OccurrenceKey, PortBinding, Submission,
)
from .engine_prepare import _discover_hbm_bus_aliases, _retrofit_hbm_bus_tasks
from .engine_types import Allocation, Event, ExecutionError, Resource, _FULL_FINITE_PREFLIGHT
from .host import cursor_for


class _EngineCatalogMixin:
    def _clear_prepared_caches(self) -> None:
        self._compiled_programs.clear()
        self._stage_id_prepared.clear()
        self._stage_proto_prepared.clear()
        self._stage_prepared.clear()
        self._stage_prepared_id.clear()
        self._pair_rate_cache.clear()

    def _discover_hbm_bus_aliases(self) -> None:
        probe = dict(self._hbm_bus_aliases)
        if not _discover_hbm_bus_aliases(self.resources, probe):
            return
        for base, bus in probe.items():
            if base not in self._hbm_bus_aliases:
                self.bind_hbm_bus(bus)

    def _flush_pending_hbm_retrofits(self) -> None:
        if not self._pending_hbm_retrofits:
            return
        pending = tuple(self._pending_hbm_retrofits.items())
        self._pending_hbm_retrofits.clear()
        for base, (bus, replacing) in pending:
            _retrofit_hbm_bus_tasks(
                self._tasks, self._native_slots, base, bus,
                replacing=replacing, include_native=True,
            )

    def bind_hbm_bus(self, bus) -> None:
        """Bind a die's scalar HBM resource onto its directional bus rows."""
        base = (bus.row_r[:-2] if bus.row_r.endswith(":r")
                and bus.row_w == bus.row_r[:-2] + ":w" else "")
        if not base or base not in self.resources:
            return
        if bus.row_r not in self.resources or bus.row_w not in self.resources:
            return
        prev = self._hbm_bus_aliases.get(base)
        if prev == bus:
            return
        replacing = prev is not None
        needs_retrofit = any(
            (task.remaining.get(base, 0.0) > 0.0
             and (replacing or (bus.row_r not in task.remaining and bus.row_w not in task.remaining)))
            or (replacing and any(item[0] == base for item in task.hbm_accounting))
            for task in self._tasks.values()
        )
        if needs_retrofit and self._accel is not None and self._accel.enabled:
            self._accel.disable(self, "mixed_frontier")
        self._hbm_bus_aliases[base] = bus
        self._hbm_bus_epoch += 1
        self._clear_prepared_caches()
        if needs_retrofit and _retrofit_hbm_bus_tasks(
            self._tasks, self._native_slots, base, bus, replacing=replacing,
        ):
            self._pending_hbm_retrofits[base] = (bus, replacing)

    def add_hbm_bus(self, bus) -> tuple[Resource, Resource]:
        """Declare a die's 2-row directional HBM bus resources on this Engine."""
        rows = (
            Resource(bus.row_r, 1.0, unit="row_fraction", policy="fair", units=1),
            Resource(bus.row_w, 1.0, unit="row_fraction", policy="fair", units=1),
        )
        adding = any(row.name not in self.resources for row in rows)
        if adding and self._accel is not None and self._accel.enabled and self._native_slots:
            self._accel.disable(self, "mixed_frontier")
        self.extend(rows, _discover_hbm=False)
        self.bind_hbm_bus(bus)
        return rows

    def new_occurrence(self, *, replica: int = 0, rank: int = 0,
                       stage: int = 0, ordinal: int = 0) -> OccurrenceKey:
        """Allocate one monotonically increasing launch identity."""
        self._next_launch += 1
        return OccurrenceKey(
            runtime_uid=self.uid, launch=self._next_launch,
            ordinal=int(ordinal), replica=int(replica), rank=int(rank),
            stage=int(stage))

    def event_for(self, occurrence: OccurrenceKey, local_id: int, *,
                  boundary: str = "done") -> Event:
        """Return the real event of one materialized catalog node."""
        cursor = self.program_cursors.get(occurrence.launch)
        if cursor is None:
            raise ExecutionError(f"occurrence_not_materialized:{occurrence.launch}")
        event = cursor["nodes"].get(int(local_id)) or cursor["promises"].get(
            int(local_id))
        if event is None:
            raise ExecutionError(f"catalog_node_missing:{local_id}")
        if boundary == "done":
            return event
        if boundary == "start":
            return self.start_boundary(event, f"{event.name}:started")
        raise ExecutionError(f"catalog_boundary_unknown:{boundary}")

    def snapshot_submission(self, submission: Submission) -> Mapping[str, Any]:
        """Immutable per-occurrence receipt; never a global service delta."""
        from types import MappingProxyType
        record = self.occurrence_metrics.get(
            str(submission.occurrence.launch), {})
        metrics = MappingProxyType({
            "submitted": int(record.get("submitted", 0)),
            "started": int(record.get("started", 0)),
            "completed": int(record.get("completed", 0)),
            "service": MappingProxyType(dict(record.get("service", {}))),
        })
        return MappingProxyType({
            "occurrence": submission.occurrence,
            "start": submission.start,
            "done": submission.done,
            "ports": MappingProxyType(dict(submission.ports)),
            "allocations": MappingProxyType(dict(submission.allocations)),
            "selected_program_id": submission.selected_program_id,
            "metrics": metrics,
        })

    @staticmethod
    def _scalar(value, binding) -> float:
        evaluate = getattr(value, "evaluate", None)
        result = float(evaluate(binding) if callable(evaluate) else value)
        if not math.isfinite(result):
            raise ExecutionError("catalog_scalar_nonfinite")
        return result

    def _resource_role(self, binding: BindingFrame, role: str) -> str:
        return str(binding.resources.get(role, role))

    def _allocation_for(self, binding: BindingFrame, spec, *, unit=None):
        occurrence = binding.occurrence
        scope = str(getattr(spec, "scope_kind", "invocation"))
        ordinal = (occurrence.ordinal if scope == "invocation"
                   else int(binding.coordinates.get(f"{scope}_ordinal", 0)))
        key = AllocationKey(
            occurrence=occurrence, scope_kind=scope, scope_ordinal=ordinal,
            symbol=int(spec.symbol),
            generation=int(getattr(spec, "generation", 0)))
        amount = self._scalar(getattr(spec, "amount", 0.0), binding)
        if amount < 0:
            raise ExecutionError(f"catalog_allocation_negative:{spec.label}")
        units = tuple(int(unit) for unit in getattr(spec, "units", (0,)))
        bound_pool = binding.pool_bindings.get(spec.pool_role)
        pool = getattr(bound_pool, "name", bound_pool) or spec.pool_role
        # A repeated finite panel keeps one occurrence but reuses the same
        # symbol; the live window coordinate must distinguish its grants.
        panel = "" if unit is None else f"u{int(unit)}:"
        name = (f"{occurrence.runtime_uid}:{occurrence.launch}:{panel}"
                f"{key.symbol}:{key.scope_ordinal}")
        return key, Allocation(
            name=name, pool=str(pool), size=amount, units=units)

    def _catalog_preflight(self, catalog: ProgramCatalog,
                           binding: BindingFrame, after) -> None:
        if not isinstance(binding, BindingFrame):
            raise ExecutionError("submit_catalog_requires_BindingFrame")
        if not isinstance(binding.occurrence, OccurrenceKey):
            raise ExecutionError("submit_catalog_requires_OccurrenceKey")
        if binding.occurrence.launch <= 0:
            raise ExecutionError("occurrence_launch_invalid")
        for predecessor in after:
            if predecessor.time is not None:
                continue
            if (predecessor.sequence not in self._tasks
                    and predecessor.sequence not in self._promises):
                raise ExecutionError(
                    f"dependency_not_submitted:{predecessor.name}")
        guard = getattr(catalog, "guard", None)
        if callable(guard) and not guard(binding):
            raise ExecutionError(f"catalog_guard_failed:{catalog.catalog_id}")
        host_spec = getattr(catalog, "host_spec", None)
        if host_spec is not None:
            role = str(getattr(host_spec, "fifo_role", ""))
            if not role:
                raise ExecutionError("host_fifo_role_missing")
            existing = self.host_fifos.get(role)
            if existing is not None and not existing.can_submit():
                raise ExecutionError(f"host_fifo_queue_full:{role}")
            work = self._scalar(getattr(host_spec, "submit_work_ms", 0.0), binding)
            if work < 0:
                raise ExecutionError("host_submit_work_negative")

        def check(stage, frame):
            condition = getattr(stage, "condition", None)
            if condition and not frame.conditions.get(condition, False):
                return
            for term in getattr(stage, "work", ()):
                role = self._resource_role(frame, term.physical_resource_role)
                if role not in self.resources:
                    raise ExecutionError(f"resource_missing:{role}")
                if self._scalar(term.issued, frame) < 0:
                    raise ExecutionError(f"catalog_work_negative:{stage.label}")
            for term in getattr(stage, "caps", ()):
                role = self._resource_role(frame, term.physical_resource_role)
                if role not in self.resources:
                    raise ExecutionError(f"resource_missing:{role}")
                if self._scalar(term.rate, frame) <= 0:
                    raise ExecutionError(f"catalog_cap_nonpositive:{stage.label}")
            if self._scalar(getattr(stage, "latency_expr", 0.0), frame) < 0:
                raise ExecutionError(f"catalog_latency_negative:{stage.label}")
            for spec in getattr(stage, "acquire", ()):
                _, allocation = self._allocation_for(frame, spec)
                if allocation.pool not in self.pools:
                    raise ExecutionError(f"pool_missing:{allocation.pool}")
                if allocation.size > self.pools[allocation.pool]:
                    raise ExecutionError(
                        f"allocation_exceeds_capacity:{allocation.name}")
            for spec in getattr(stage, "placement", ()):
                role = self._resource_role(frame, spec.physical_resource_role)
                if role not in self.resources:
                    raise ExecutionError(f"resource_missing:{role}")
                if any(unit < 0 or unit >= self.resources[role].units
                       for unit in spec.units):
                    raise ExecutionError(f"resource_unit_out_of_range:{role}")
            nested = getattr(stage, "nested_catalog", None)
            if nested is not None:
                self._catalog_preflight(nested, frame, ())

        finite = getattr(catalog, "finite", None)
        if finite is not None:
            if not isinstance(finite.total, int) or finite.total < 0:
                raise ExecutionError("finite_total_invalid")
            classes = {int(item.class_id): item.stages for item in finite.classes}
            if not classes and finite.total:
                raise ExecutionError("finite_classes_missing")
            # Preflight is bounded by the declared live window. A million-panel
            # producer must not pay a million `index_of` calls before its first
            # event; class/coordinate validation continues at each activation.
            window = self._finite_window(finite)
            probe = finite.total if finite.total <= _FULL_FINITE_PREFLIGHT \
                else min(finite.total, window)
            for index in range(probe):
                class_id, _, _, _, _ = finite.index_of(index)
                if int(class_id) not in classes:
                    raise ExecutionError(f"finite_class_missing:{class_id}")
                frame = self._unit_binding(binding, index, finite)
                for stage in classes[int(class_id)]:
                    check(stage, frame)
            for stage in tuple(finite.setup or ()) + tuple(finite.drain or ()):
                check(stage, binding)
            return
        for stage in catalog.stages:
            check(stage, binding)

    @staticmethod
    def _finite_window(finite: FiniteProgramSpec) -> int:
        """Maximum live panels for one repeated catalog (declared slots)."""
        slots = getattr(finite, "slots", 1)
        if not isinstance(slots, int) or isinstance(slots, bool):
            raise ExecutionError("finite_slots_invalid")
        total = int(finite.total)
        if total <= 0:
            return 0
        return max(1, min(total, max(1, slots)))

    @staticmethod
    def _unit_binding(binding: BindingFrame, index: int,
                      finite: FiniteProgramSpec) -> BindingFrame:
        class_id, end, output, panel, slot = finite.index_of(index)
        coordinates = dict(binding.coordinates)
        coordinates.update({
            "index": int(index), "class_id": int(class_id),
            "end_exclusive": int(end), "output_ordinal": int(output),
            "panel_ordinal": int(panel), "slot_ordinal": int(slot),
        })
        from types import MappingProxyType
        return replace(binding, coordinates=MappingProxyType(coordinates))

    def _submit_catalog_node(self, stage: StageSpec, frame: BindingFrame, *,
                             name: str, deps, owner: str, register, ports,
                             allocations, effects, unit=None, acquired=None):
        condition = getattr(stage, "condition", None)
        if condition and not frame.conditions.get(condition, False):
            event = self.submit(
                name, after=tuple(deps), owner=owner,
                on_complete=effects.get("resolve"))
            register(stage, event)
            return event
        work = {}
        for term in getattr(stage, "work", ()):
            role = self._resource_role(frame, term.physical_resource_role)
            amount = self._scalar(term.issued, frame) * float(
                getattr(term, "service_conversion", 1.0))
            if amount:
                work[role] = work.get(role, 0.0) + amount
        caps = {}
        for term in getattr(stage, "caps", ()):
            caps[self._resource_role(frame, term.physical_resource_role)] = (
                self._scalar(term.rate, frame))
        placement = {}
        for spec in getattr(stage, "placement", ()):
            role = self._resource_role(frame, spec.physical_resource_role)
            placement[role] = tuple(spec.units)
        acquire = []
        for spec in getattr(stage, "acquire", ()):
            key, allocation = self._allocation_for(frame, spec, unit=unit)
            acquire.append(allocation)
            if acquired is not None:
                # A release may name a symbol acquired by an earlier stage of
                # the same catalog/panel; identity is the live grant, not the
                # stage-local acquire tuple.
                acquired[int(key.symbol)] = allocation
            allocations[str(key.symbol)] = key
        free = []
        for spec in getattr(stage, "release", ()):
            allocation = None
            if acquired is not None:
                allocation = acquired.pop(int(spec.symbol), None)
            if allocation is not None:
                free.append(allocation)
        state_effects = tuple(getattr(stage, "state_effects", ()) or ())
        on_complete = effects.get("resolve")
        if state_effects:
            def commit(_event, effects=state_effects, callback=on_complete,
                       frame=frame):
                active = tuple(
                    effect for effect in effects
                    if not getattr(effect, "condition", "")
                    or frame.conditions.get(effect.condition, False))
                self._apply_state_effects(active, _event)
                if callback is not None:
                    callback(_event)
            on_complete = commit
        event = self.submit(
            name, work=work or None,
            latency=self._scalar(getattr(stage, "latency_expr", 0.0), frame),
            caps=caps or None, acquire=tuple(acquire), free=tuple(free),
            placement=placement or None,
            coupled=bool(getattr(stage, "coupled", False)),
            priority=int(getattr(stage, "priority", 0)),
            cohort_size=int(getattr(stage, "cohort_size", 1)),
            cohort_widened=tuple(getattr(stage, "cohort_widened", ())) or None,
            after=tuple(deps), owner=owner, on_complete=on_complete,
            metadata={"occurrence": frame.occurrence,
                      "catalog_stage": stage.label})
        register(stage, event)
        for port in getattr(stage, "output_ports", ()):
            if port in ports:
                raise ExecutionError(f"catalog_port_redefined:{port}")
            ports[port] = PortBinding(
                ready=event, producer=frame.occurrence,
                access="write", output=port)
        return event

    def _apply_state_effects(self, effects, event: Event) -> None:
        """Commit declared state transitions into the one runtime ledger."""
        if not effects:
            return
        ledger = self.ensure_state_ledger()
        at = event.time if event.time is not None else self.now
        ledger.apply(effects, at=at, sequence=event.sequence)
        # Keep the compact per-target commit history as a projection only.
        for effect in effects:
            if effect.action == "commit":
                self.state_versions.setdefault(
                    effect.target, []).append(effect.version)
            elif effect.action == "rollback":
                versions = self.state_versions.get(effect.target, [])
                if versions and versions[-1] == effect.version:
                    versions.pop()

    def submit_catalog(self, catalog: ProgramCatalog, binding: BindingFrame, *,
                       name: str, after: Sequence[Event] = (),
                       owner: str = "") -> Submission:
        """Submit one immutable catalog through the unique event kernel.

        Flat catalogs publish every declared stage as a relative DAG. Repeated
        catalogs publish at most ``finite.slots`` panels and issue the next
        absolute index only when a live panel retires, so their live frontier
        is independent of ``finite.total``. Validation happens before the first
        published event; for repeated catalogs larger than the full-preflight
        bound only the declared live window is preflighted and each remaining
        panel's class/coordinates are validated at its activation.
        """
        after = tuple(after)
        self._catalog_preflight(catalog, binding, after)
        occurrence = binding.occurrence
        self._next_launch = max(self._next_launch, occurrence.launch)
        start = self.submit(f"{name}:start", after=after, owner=owner)
        device_root = start
        host_spec = getattr(catalog, "host_spec", None)
        if host_spec is not None:
            cursor = cursor_for(
                self, host_spec.fifo_role, occurrence.runtime_uid, host_spec)
            sequence = cursor.note_submit()
            host_work = self._scalar(
                getattr(host_spec, "submit_work_ms", 0.0), binding)
            role = str(host_spec.fifo_role)
            if host_work > 0 and role in self.resources:
                device_root = self.submit(
                    f"{name}:host", work={role: host_work}, after=(start,),
                    owner=owner, metadata={"occurrence": occurrence},
                    on_complete=lambda _event, c=cursor, s=sequence:
                    c.note_host_done(s))
            else:
                device_root = self.submit(
                    f"{name}:host", latency=host_work, after=(start,),
                    owner=owner, metadata={"occurrence": occurrence},
                    on_complete=lambda _event, c=cursor, s=sequence:
                    c.note_host_done(s))
        ports: dict[str, PortBinding] = {}
        allocations: dict[str, AllocationKey] = {}
        nodes: dict[int, Event] = {}
        promises: dict[int, Event] = {}
        cursor = {
            "nodes": nodes, "promises": promises, "ports": ports,
            "allocations": allocations,
            "catalog_id": getattr(catalog, "catalog_id", ""),
            "next": 0, "finished": 0, "live": set(), "closed": False,
            "previous": {}, "setup": (),
        }
        self.program_cursors[occurrence.launch] = cursor

        def register(stage, event):
            nodes[int(getattr(stage, "local_id", len(nodes)))] = event

        def stage_events(stages, frame, prefix, roots, acquired=None):
            created = []
            for stage in stages:
                local = int(getattr(stage, "local_id", len(nodes)))
                promise = self.future(f"{name}:{prefix}:{local}:node")
                promises[local] = promise
                deps = list(roots)
                for edge in getattr(stage, "relative_after", ()):
                    if edge.source_node is not None:
                        source = int(edge.source_node)
                        dep = promises.get(source) or nodes.get(source)
                        if dep is not None:
                            deps.append(dep)
                    elif edge.external_port:
                        external = frame.values.get(edge.external_port)
                        if external is not None:
                            deps.append(external.ready)
                nested = getattr(stage, "nested_catalog", None)
                if nested is not None:
                    # Nested work is a child submission of the same runtime and
                    # must inherit the parent stage's real dependencies; it may
                    # never start before the declared ready boundary.
                    sub = self.submit_catalog(
                        nested, replace(
                            frame, occurrence=self.new_occurrence(
                                replica=frame.occurrence.replica,
                                rank=frame.occurrence.rank,
                                stage=frame.occurrence.stage)),
                        name=f"{name}:{prefix}:{local}:nested", owner=owner,
                        after=tuple(dict.fromkeys(deps)))
                    deps.append(sub.done)
                created.append(self._submit_catalog_node(
                    stage, frame, name=f"{name}:{prefix}:{stage.label}",
                    deps=tuple(dict.fromkeys(deps)), owner=owner,
                    register=register, ports=ports, allocations=allocations,
                    effects={"resolve": (lambda _event, p=promise: self.resolve(p))},
                    acquired=acquired))
            return tuple(created)

        finite = getattr(catalog, "finite", None)
        if finite is None:
            acquired: dict[int, Allocation] = {}
            sinks = {int(getattr(stage, "local_id", -1))
                     for stage in catalog.stages}
            for stage in catalog.stages:
                for edge in getattr(stage, "relative_after", ()):
                    if edge.source_node is not None:
                        sinks.discard(int(edge.source_node))
            stage_events(catalog.stages, binding, "s", (device_root,),
                         acquired=acquired)
            tails = [nodes[local] for local in sorted(sinks) if local in nodes]
            done = self.submit(f"{name}:done", after=tuple(dict.fromkeys(tails)),
                               owner=owner)
        else:
            done = self._submit_finite_catalog(
                catalog, finite, binding, name=name, owner=owner,
                device_root=device_root, cursor=cursor,
                stage_events=stage_events)
        return Submission(
            occurrence=occurrence, start=start, done=done,
            ports=ports, allocations=allocations,
            selected_program_id=str(getattr(catalog, "catalog_id", "")))

    def _submit_finite_catalog(self, catalog, finite, binding, *, name, owner,
                               device_root, cursor, stage_events):
        """Materialize one bounded live window of a repeated class catalog.

        At most ``finite.slots`` panels carry tasks at any time. Each retired
        panel issues the next absolute index, so the live frontier is
        O(slots * class stages) and independent of ``finite.total``; no Event,
        promise or task exists for an inactive panel. Setup/drain/minimum
        boundaries keep their declared causal position.
        """
        nodes = cursor["nodes"]
        ports = cursor["ports"]
        allocations = cursor["allocations"]
        invocation_acquired: dict[int, Allocation] = {}
        setup = tuple(stage_events(
            finite.setup or (), binding, "setup", (device_root,),
            acquired=invocation_acquired))
        cursor["setup"] = setup
        classes = {int(item.class_id): item.stages for item in finite.classes}
        total = int(finite.total)
        window = self._finite_window(finite)
        minimum = self._scalar(getattr(finite, "minimum_ms", 0.0), binding)
        # A minimum invocation duration starts with the invocation, never after
        # the last panel's service, so it is published with the launch.
        minimum_event = None
        if minimum > 0:
            minimum_event = self.submit(
                f"{name}:minimum", latency=minimum, after=(device_root,),
                owner=owner)
        done = self.future(f"{name}:done")

        def issue(index: int) -> None:
            class_id, _, _, _, _ = finite.index_of(index)
            stages = classes.get(int(class_id))
            if stages is None:
                raise ExecutionError(f"finite_class_missing:{class_id}")
            frame = self._unit_binding(binding, index, finite)
            roots = list(setup) or [device_root]
            acquired = dict(invocation_acquired)
            # A previous-iteration edge must resolve against the snapshot taken
            # before this tile is issued; same-tile ``after`` edges resolve
            # against the nodes already published by this tile.
            previous = dict(cursor["previous"])
            tails = []
            for stage in stages:
                local = int(getattr(stage, "local_id", len(nodes)))
                deps = list(roots)
                for edge in getattr(stage, "relative_after", ()):
                    if edge.source_node is not None:
                        source = int(edge.source_node)
                        if int(getattr(edge, "iteration_delta", 0)) < 0:
                            dep = previous.get(source)
                        else:
                            dep = nodes.get(source)
                        if dep is not None:
                            deps.append(dep)
                    elif edge.external_port:
                        external = frame.values.get(edge.external_port)
                        if external is not None:
                            deps.append(external.ready)
                if local in previous:
                    deps.append(previous[local])
                nested = getattr(stage, "nested_catalog", None)
                if nested is not None:
                    sub = self.submit_catalog(
                        nested, replace(
                            frame, occurrence=self.new_occurrence(
                                replica=frame.occurrence.replica,
                                rank=frame.occurrence.rank,
                                stage=frame.occurrence.stage)),
                        name=f"{name}:u{index}:{local}:nested", owner=owner,
                        after=tuple(dict.fromkeys(deps)))
                    deps.append(sub.done)
                event = self._submit_catalog_node(
                    stage, frame, name=f"{name}:u{index}:{stage.label}",
                    deps=tuple(dict.fromkeys(deps)), owner=owner,
                    register=lambda _stage, _event, _local=local:
                        nodes.__setitem__(_local, _event),
                    ports=ports, allocations=allocations, effects={},
                    unit=index, acquired=acquired)
                cursor["previous"][local] = event
                nodes[local] = event
                tails.append(event)
            self.submit(
                f"{name}:u{index}:retire", after=tuple(tails) or tuple(roots),
                owner=owner,
                on_complete=lambda _event, _index=index: retire(_index))

        def retire(index: int) -> None:
            cursor["finished"] += 1
            cursor["live"].discard(index)
            if cursor["finished"] >= total:
                close()
                return
            while cursor["next"] < total and len(cursor["live"]) < window:
                nxt = cursor["next"]
                cursor["next"] = nxt + 1
                cursor["live"].add(nxt)
                issue(nxt)

        def close() -> None:
            if cursor["closed"]:
                return
            cursor["closed"] = True
            tails = []
            drain = tuple(finite.drain or ())
            if drain:
                tails.extend(stage_events(
                    drain, binding, "drain", tuple(setup) or (device_root,),
                    acquired=dict(invocation_acquired)))
            if minimum_event is not None:
                tails.append(minimum_event)
            if tails:
                self.submit(
                    f"{name}:done_boundary",
                    after=tuple(dict.fromkeys(tails)), owner=owner,
                    on_complete=lambda _event: self.resolve(done))
            else:
                self.resolve(done)

        if total <= 0:
            # A zero-panel catalog still executes its real setup/drain work.
            close()
        else:
            for _ in range(min(window, total)):
                index = cursor["next"]
                cursor["next"] = index + 1
                cursor["live"].add(index)
                issue(index)
        return done
