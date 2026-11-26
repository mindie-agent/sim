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
import heapq
import math
from typing import Any, Mapping, Sequence

from .contracts import (
    AllocationKey, BindingFrame, OccurrenceKey, PortBinding, Submission,
)
from .engine_prepare import _discover_hbm_bus_aliases, _retrofit_hbm_bus_tasks
from .engine_types import Allocation, Event, ExecutionError, Resource, _FULL_FINITE_PREFLIGHT
from .host import cursor_for
from .memory_physics import HbmBusSpec

_HBM_ROW_READ_SUFFIX = ":r"
_HBM_ROW_WRITE_SUFFIX = ":w"
_HBM_DEFAULT_READ_PREFIX = "hbm_bus_r@d"
_HBM_DEFAULT_WRITE_PREFIX = "hbm_bus_w@d"
_UNIT_ROW_RATE = 1.0


class _EngineCatalogMixin:
    def _clear_prepared_caches(self) -> None:
        self._compiled_programs.clear()
        self._stage_id_prepared.clear()
        self._stage_proto_prepared.clear()
        self._stage_prepared.clear()
        self._stage_prepared_id.clear()
        self._pair_rate_cache.clear()
        self._seen_floor_peaks.clear()
        self._validated_placements.clear()

    def _hbm_bus_registry(self) -> tuple[dict[tuple[str, str], tuple[Any, str | None]], set[str]]:
        declared = self.__dict__.get("_declared_hbm_buses")
        if declared is None:
            declared = {}
            self.__dict__["_declared_hbm_buses"] = declared
        explicit = self.__dict__.get("_explicit_hbm_bases")
        if explicit is None:
            explicit = set()
            self.__dict__["_explicit_hbm_bases"] = explicit
        return declared, explicit

    def _resolve_hbm_bus_base(self, bus, base: str | None = None) -> str:
        """Resolve the scalar HBM resource name governed by directional ``bus``."""
        if bus.row_r == bus.row_w:
            return ""
        if base is not None:
            return (base if base in self.resources
                    and base not in (bus.row_r, bus.row_w) else "")
        if (bus.row_r.endswith(_HBM_ROW_READ_SUFFIX)
                and bus.row_w == bus.row_r[:-2] + _HBM_ROW_WRITE_SUFFIX):
            cand = bus.row_r[:-2]
            return cand if cand in self.resources else ""
        die = int(bus.die)
        for cand in (f"hbm@d{die}", f"memory@d{die}"):
            if cand in self.resources and cand not in (bus.row_r, bus.row_w):
                return cand
        if "hbm" in self.resources and "hbm" not in (bus.row_r, bus.row_w):
            declared, explicit_bases = self._hbm_bus_registry()
            prev = self._hbm_bus_aliases.get("hbm")
            if "hbm" in explicit_bases:
                return "hbm" if (prev is not None and int(prev.die) == die) else ""
            if die == 0:
                return "hbm"
            has_scoped = any(
                r.startswith(("hbm@d", "memory@d")) for r in self.resources
            )
            has_die0 = any(
                int(spec.die) == 0 and exp_base is None
                for spec, exp_base in declared.values()
            ) or (
                f"{_HBM_DEFAULT_READ_PREFIX}0" in self.resources
                and f"{_HBM_DEFAULT_WRITE_PREFIX}0" in self.resources
            )
            if has_scoped or has_die0:
                return ""
            if prev is not None:
                return "hbm" if int(prev.die) == die else ""
            return "hbm"
        return ""

    def _discover_hbm_bus_aliases(self) -> None:
        declared, explicit_bases = self._hbm_bus_registry()
        if "hbm" in self._hbm_bus_aliases and "hbm" not in explicit_bases:
            cur_hbm = self._hbm_bus_aliases["hbm"]
            if self._resolve_hbm_bus_base(cur_hbm) != "hbm":
                del self._hbm_bus_aliases["hbm"]
                self._hbm_bus_epoch += 1
                self._clear_prepared_caches()
        for bus, exp_base in sorted(
            declared.values(),
            key=lambda item: (item[1] is None, int(item[0].die) != 0, int(item[0].die)),
        ):
            if bus.row_r in self.resources and bus.row_w in self.resources:
                self._bind_hbm_bus_impl(
                    bus, base=exp_base, record=False, explicit=(exp_base is not None),
                )
        probe = dict(self._hbm_bus_aliases)
        found = _discover_hbm_bus_aliases(self.resources, probe)
        for name, res_r in self.resources.items():
            if not name.startswith(_HBM_DEFAULT_READ_PREFIX) or res_r.rate != _UNIT_ROW_RATE:
                continue
            die_str = name[len(_HBM_DEFAULT_READ_PREFIX):]
            if not die_str.isdigit():
                continue
            row_w = f"{_HBM_DEFAULT_WRITE_PREFIX}{die_str}"
            if (name, row_w) in declared:
                continue
            res_w = self.resources.get(row_w)
            if res_w is None or res_w.rate != _UNIT_ROW_RATE:
                continue
            die = int(die_str)
            probe_bus = HbmBusSpec(die, _UNIT_ROW_RATE, _UNIT_ROW_RATE, 0.0,
                                   row_r=name, row_w=row_w)
            base = self._resolve_hbm_bus_base(probe_bus)
            if not base or base in probe:
                continue
            base_rate = float(self.resources[base].rate)
            if base_rate <= 0.0:
                continue
            probe[base] = HbmBusSpec(die, base_rate, base_rate, 0.0,
                                     row_r=name, row_w=row_w)
            found = True
        if not found:
            return
        for base, bus in probe.items():
            if base not in self._hbm_bus_aliases:
                self._bind_hbm_bus_impl(bus, base=base, record=False, explicit=False)

    def _flush_pending_hbm_retrofits(self) -> None:
        if not self._pending_hbm_retrofits:
            return
        pending = tuple(self._pending_hbm_retrofits.items())
        self._pending_hbm_retrofits.clear()
        for base, (bus, replacing) in pending:
            _retrofit_hbm_bus_tasks(
                self._tasks, self._native_slots, base, bus,
                engine=self, replacing=replacing, include_native=True,
            )

    def _bind_hbm_bus_impl(
        self, bus, *, base: str | None = None, record: bool = True, explicit: bool = False,
    ) -> None:
        if bus.row_r not in self.resources or bus.row_w not in self.resources:
            return
        declared, explicit_bases = self._hbm_bus_registry()
        if record:
            declared[(bus.row_r, bus.row_w)] = (bus, base)
        resolved = self._resolve_hbm_bus_base(bus, base)
        if not resolved:
            return
        if explicit:
            explicit_bases.add(resolved)
        prev = self._hbm_bus_aliases.get(resolved)
        if prev == bus:
            return
        replacing = prev is not None
        needs_retrofit = any(
            (task.remaining.get(resolved, 0.0) > 0.0
             and (replacing or (bus.row_r not in task.remaining and bus.row_w not in task.remaining)))
            or (replacing and any(item[0] == resolved for item in task.hbm_accounting))
            for task in self._tasks.values()
        )
        if needs_retrofit and self._accel is not None and self._accel.enabled:
            self._accel.disable(self, "mixed_frontier")
        self._hbm_bus_aliases[resolved] = bus
        self._hbm_bus_epoch += 1
        self._clear_prepared_caches()
        if needs_retrofit and _retrofit_hbm_bus_tasks(
            self._tasks, self._native_slots, resolved, bus,
            engine=self, replacing=replacing,
        ):
            self._pending_hbm_retrofits[resolved] = (bus, replacing)

    def bind_hbm_bus(self, bus, *, base: str | None = None) -> None:
        """Bind a die's scalar HBM resource onto its directional bus rows."""
        self._bind_hbm_bus_impl(bus, base=base, record=True, explicit=(base is not None))

    def add_hbm_bus(self, bus, *, base: str | None = None) -> tuple[Resource, Resource]:
        """Declare a die's 2-row directional HBM bus resources on this Engine."""
        declared, _ = self._hbm_bus_registry()
        declared[(bus.row_r, bus.row_w)] = (bus, base)
        rows = (
            Resource(bus.row_r, _UNIT_ROW_RATE, unit="row_fraction", policy="fair", units=1),
            Resource(bus.row_w, _UNIT_ROW_RATE, unit="row_fraction", policy="fair", units=1),
        )
        adding = any(row.name not in self.resources for row in rows)
        if adding and self._accel is not None and self._accel.enabled and self._native_slots:
            self._accel.disable(self, "mixed_frontier")
        self.extend(rows, _discover_hbm=False)
        self._bind_hbm_bus_impl(bus, base=base, record=True, explicit=(base is not None))
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
            starts = cursor.get("starts")
            target = starts.get(int(local_id), event) if starts else event
            return self.start_boundary(target, f"{event.name}:started")
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

    def _host_fifo_role(self, binding: BindingFrame, role: str) -> str:
        if binding.host_fifo:
            return self._resource_role(binding, str(binding.host_fifo))
        return self._resource_role(binding, role)

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

    @staticmethod
    def _stage_active(stage, frame: BindingFrame) -> bool:
        condition = getattr(stage, "condition", None)
        return not condition or bool(frame.conditions.get(condition, False))

    @staticmethod
    def _order_catalog_stages(stages: Sequence) -> tuple[tuple[int, Any], ...]:
        """Return stages in deterministic topological order by intra-group edges."""
        if not stages:
            return ()
        id_to_pos: dict[int, int] = {}
        local_ids: list[int] = []
        for pos, stage in enumerate(stages):
            local = int(getattr(stage, "local_id", pos))
            if local in id_to_pos:
                raise ExecutionError(f"catalog_duplicate_node:{local}")
            id_to_pos[local] = pos
            local_ids.append(local)
        n_stages = len(stages)
        indegree = [0] * n_stages
        succs: list[list[int]] = [[] for _ in range(n_stages)]
        for dst, stage in enumerate(stages):
            seen_src: set[int] = set()
            for edge in getattr(stage, "relative_after", ()):
                if (edge.source_node is None
                        or int(getattr(edge, "iteration_delta", 0)) != 0):
                    continue
                src = id_to_pos.get(int(edge.source_node))
                if src is not None and src not in seen_src:
                    seen_src.add(src)
                    succs[src].append(dst)
                    indegree[dst] += 1
        ready = [pos for pos in range(n_stages) if indegree[pos] == 0]
        heapq.heapify(ready)
        ordered: list[tuple[int, Any]] = []
        while ready:
            pos = heapq.heappop(ready)
            ordered.append((local_ids[pos], stages[pos]))
            for nxt in succs[pos]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(ready, nxt)
        if len(ordered) != n_stages:
            raise ExecutionError("catalog_dependency_cycle")
        return tuple(ordered)

    def _resolve_edge_dep(
        self, dep: Event, edge, *, start_event: Event | None = None,
    ) -> Event:
        boundary = str(getattr(edge, "source_boundary", "done") or "done")
        if boundary == "done":
            return dep
        if boundary == "start":
            target = start_event if start_event is not None else dep
            if target.started is None and target.sequence not in self._tasks:
                if (any(target is obs for waiters in self._start_waiters.values() for obs in waiters)
                        or any(task.start_observer is target for task in self._tasks.values())):
                    return target
            return self.start_boundary(target, f"{dep.name}:started")
        raise ExecutionError(f"catalog_boundary_unknown:{boundary}")

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
            phys_role = self._host_fifo_role(binding, role)
            existing = self.host_fifos.get(phys_role)
            if existing is not None:
                if not existing.can_submit():
                    raise ExecutionError(f"host_fifo_queue_full:{role}")
            else:
                cap = getattr(host_spec, "queue_capacity", None)
                if cap is not None and cap <= 0:
                    raise ExecutionError(f"host_fifo_queue_full:{role}")
            work = self._scalar(getattr(host_spec, "submit_work_ms", 0.0), binding)
            if work < 0:
                raise ExecutionError("host_submit_work_negative")
            for port_name in tuple(getattr(host_spec, "host_dependency_ports", ()) or ()):
                port_binding = binding.values.get(port_name)
                if port_binding is None:
                    continue
                dep = getattr(port_binding, "ready", port_binding)
                if dep.time is not None:
                    continue
                if dep.sequence not in self._tasks and dep.sequence not in self._promises:
                    raise ExecutionError(f"dependency_not_submitted:{dep.name}")

        for port_spec in getattr(catalog, "ports", ()):
            boundary = str(getattr(port_spec, "producer_boundary", "done") or "done")
            if boundary not in ("done", "start"):
                raise ExecutionError(f"catalog_boundary_unknown:{boundary}")

        def check(stage, frame):
            if not self._stage_active(stage, frame):
                return
            for edge in getattr(stage, "relative_after", ()):
                boundary = str(getattr(edge, "source_boundary", "done") or "done")
                if boundary not in ("done", "start"):
                    raise ExecutionError(f"catalog_boundary_unknown:{boundary}")
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
            for counter_id, amount_expr in getattr(stage, "deliveries", ()):
                if not counter_id:
                    raise ExecutionError(f"catalog_delivery_counter_missing:{stage.label}")
                if self._scalar(amount_expr, frame) < 0:
                    raise ExecutionError(f"catalog_delivery_negative:{stage.label}")
            for spec in getattr(stage, "acquire", ()):
                if not self._stage_active(spec, frame):
                    continue
                _, allocation = self._allocation_for(frame, spec)
                if allocation.pool not in self.pools:
                    raise ExecutionError(f"pool_missing:{allocation.pool}")
                if allocation.size > self.pools[allocation.pool]:
                    raise ExecutionError(
                        f"allocation_exceeds_capacity:{allocation.name}")
            for spec in getattr(stage, "release", ()):
                if not self._stage_active(spec, frame):
                    continue
                reader_port = getattr(spec, "last_reader_port", None)
                if reader_port:
                    port_binding = frame.values.get(reader_port)
                    if port_binding is not None:
                        dep = getattr(port_binding, "ready", port_binding)
                        if (dep.time is None
                                and dep.sequence not in self._tasks
                                and dep.sequence not in self._promises):
                            raise ExecutionError(f"dependency_not_submitted:{dep.name}")
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
            if self._scalar(getattr(finite, "setup_ms", 0.0), binding) < 0:
                raise ExecutionError("finite_setup_negative")
            if self._scalar(getattr(finite, "minimum_ms", 0.0), binding) < 0:
                raise ExecutionError("finite_minimum_negative")
            classes = {int(item.class_id): item.stages for item in finite.classes}
            if not classes and finite.total:
                raise ExecutionError("finite_classes_missing")
            for class_stages in classes.values():
                self._order_catalog_stages(class_stages)
            self._order_catalog_stages(tuple(finite.setup or ()))
            self._order_catalog_stages(tuple(finite.drain or ()))
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
        self._order_catalog_stages(catalog.stages)
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

    def _flush_pending_releases(
        self, pending_releases: list, ports: Mapping[str, PortBinding], *,
        only_resolved: bool = False,
    ) -> None:
        """Emit deferred allocation releases once forward internal reader ports are materialized."""
        remaining: list = []
        for entry in pending_releases:
            event, spec, allocation, reader_port, name, owner, frame, stage = entry
            binding = ports.get(reader_port)
            if binding is None and only_resolved:
                remaining.append(entry)
                continue
            reader_ev = getattr(binding, "ready", binding) if binding is not None else None
            after_ev = (
                (event, reader_ev)
                if (reader_ev is not None and reader_ev.time is None)
                else (event,)
            )
            self.submit(
                f"{name}:release:{spec.symbol}",
                after=after_ev,
                free=(allocation,),
                owner=owner,
                metadata={"occurrence": frame.occurrence, "catalog_stage": stage.label},
            )
        pending_releases[:] = remaining

    def _submit_catalog_node(self, stage: StageSpec, frame: BindingFrame, *,
                             name: str, deps, owner: str, register, ports,
                             allocations, effects, unit=None, acquired=None,
                             port_specs: Mapping[str, Any] | None = None,
                             start_event: Event | None = None,
                             pending_releases: list | None = None,
                             declared_output_ports: Sequence[str] = ()):
        if not self._stage_active(stage, frame):
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
            if not self._stage_active(spec, frame):
                continue
            key, allocation = self._allocation_for(frame, spec, unit=unit)
            acquire.append(allocation)
            if acquired is not None:
                # A release may name a symbol acquired by an earlier stage of
                # the same catalog/panel; identity is the live grant, not the
                # stage-local acquire tuple.
                acquired[int(key.symbol)] = allocation
            allocations[str(key.symbol)] = key
        free = []
        deferred_releases = []
        unresolved_releases = []
        for spec in getattr(stage, "release", ()):
            if not self._stage_active(spec, frame):
                continue
            allocation = None
            if acquired is not None:
                allocation = acquired.pop(int(spec.symbol), None)
            if allocation is None:
                continue
            reader_port = getattr(spec, "last_reader_port", None)
            if reader_port:
                binding = frame.values.get(reader_port)
                if binding is None:
                    binding = ports.get(reader_port)
                if binding is not None:
                    reader_ev = getattr(binding, "ready", binding)
                    if reader_ev is not None and reader_ev.time is None:
                        deferred_releases.append((spec, allocation, reader_ev))
                    else:
                        free.append(allocation)
                    continue
                if (pending_releases is not None
                        and port_specs is not None
                        and reader_port in port_specs
                        and getattr(port_specs[reader_port], "direction", "in") == "out"):
                    unresolved_releases.append((spec, allocation, reader_port))
                    continue
            free.append(allocation)
        deliveries = tuple(
            (str(counter_id), amount)
            for counter_id, amount_expr in getattr(stage, "deliveries", ())
            if (amount := self._scalar(amount_expr, frame)) > 0.0
        )
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
            deliveries=deliveries,
            latency_from_first_service=bool(
                getattr(stage, "latency_from_first_service", False)),
            metadata={"occurrence": frame.occurrence,
                      "catalog_stage": stage.label})
        register(stage, event)
        for spec, allocation, reader_ev in deferred_releases:
            self.submit(
                f"{name}:release:{spec.symbol}",
                after=(event, reader_ev),
                free=(allocation,),
                owner=owner,
                metadata={"occurrence": frame.occurrence,
                          "catalog_stage": stage.label},
            )
        if pending_releases is not None:
            for spec, allocation, reader_port in unresolved_releases:
                pending_releases.append(
                    (event, spec, allocation, reader_port, name, owner, frame, stage)
                )
        stage_output_ports = (
            tuple(dict.fromkeys((*getattr(stage, "output_ports", ()), *declared_output_ports)))
            if declared_output_ports else getattr(stage, "output_ports", ())
        )
        for port in stage_output_ports:
            if port in ports:
                raise ExecutionError(f"catalog_port_redefined:{port}")
            port_spec = port_specs.get(port) if port_specs else None
            boundary = str(getattr(port_spec, "producer_boundary", "done") or "done")
            if boundary == "done":
                ready_event = event
            elif boundary == "start":
                target = start_event if start_event is not None else event
                ready_event = self.start_boundary(target, f"{event.name}:started")
            else:
                raise ExecutionError(f"catalog_boundary_unknown:{boundary}")
            ports[port] = PortBinding(
                ready=ready_event, producer=frame.occurrence,
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

    def _submit_catalog_host_launch(
        self, host_spec, binding: BindingFrame, *, name: str,
        start: Event, owner: str,
    ) -> Event:
        occurrence = binding.occurrence
        phys_role = self._host_fifo_role(binding, str(host_spec.fifo_role))
        host_cursor = cursor_for(self, phys_role, occurrence.runtime_uid, host_spec)
        sequence = host_cursor.note_submit()
        host_work = self._scalar(
            getattr(host_spec, "submit_work_ms", 0.0), binding)
        host_deps = [start]
        for port_name in tuple(getattr(host_spec, "host_dependency_ports", ()) or ()):
            port_binding = binding.values.get(port_name)
            if port_binding is not None:
                host_deps.append(getattr(port_binding, "ready", port_binding))
        if host_cursor.tail_event is not None:
            host_deps.append(host_cursor.tail_event)
        after_host = tuple(dict.fromkeys(host_deps))
        on_done = (
            lambda _event, c=host_cursor, s=sequence: c.note_host_done(s, _event)
        )
        if host_work > 0.0 and phys_role in self.resources:
            launch = self.submit(
                f"{name}:host", work={phys_role: host_work}, after=after_host,
                owner=owner, metadata={"occurrence": occurrence},
                on_complete=on_done)
        else:
            launch = self.submit(
                f"{name}:host", latency=host_work, after=after_host,
                owner=owner, metadata={"occurrence": occurrence},
                on_complete=on_done)
        host_cursor.tail_event = launch
        return launch

    def submit_catalog(self, catalog: ProgramCatalog, binding: BindingFrame, *,
                       name: str, after: Sequence[Event] = (),
                       owner: str = "") -> Submission:
        """Submit one immutable catalog through the unique event kernel.

        [Cohesive-Pipeline] Single-pass catalog submission binding flat DAGs and finite repeated panels.

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
            device_root = self._submit_catalog_host_launch(
                host_spec, binding, name=name, start=start, owner=owner)
        port_specs = {
            spec.name: spec for spec in getattr(catalog, "ports", ())
        }
        out_ports_by_node: dict[int, list[str]] = {}
        if getattr(catalog, "finite", None) is None:
            for spec in getattr(catalog, "ports", ()):
                if (getattr(spec, "direction", "in") == "out"
                        and getattr(spec, "producer_node", None) is not None):
                    out_ports_by_node.setdefault(int(spec.producer_node), []).append(spec.name)
        ports: dict[str, PortBinding] = {}
        allocations: dict[str, AllocationKey] = {}
        nodes: dict[int, Event] = {}
        starts: dict[int, Event] = {}
        promises: dict[int, Event] = {}
        cursor = {
            "nodes": nodes, "starts": starts, "promises": promises, "ports": ports,
            "allocations": allocations,
            "catalog_id": getattr(catalog, "catalog_id", ""),
            "next": 0, "finished": 0, "live": set(), "closed": False,
            "previous": {}, "previous_starts": {}, "setup": (),
        }
        self.program_cursors[occurrence.launch] = cursor

        def register(stage, event):
            nodes[int(getattr(stage, "local_id", len(nodes)))] = event

        def stage_events(stages, frame, prefix, roots, acquired=None,
                         previous=None, previous_starts=None,
                         panel_history=None, panel_start_history=None,
                         panel_index=0, pending_out=None, declared_ports=None):
            created = []
            pending_releases: list = [] if pending_out is None else pending_out
            ports_by_node = out_ports_by_node if declared_ports is None else declared_ports
            group_nodes: dict[int, Event] = {}
            group_starts: dict[int, Event] = {}
            for local, stage in self._order_catalog_stages(stages):
                deps = list(roots)
                for edge in getattr(stage, "relative_after", ()):
                    if edge.source_node is not None:
                        source = int(edge.source_node)
                        delta = int(getattr(edge, "iteration_delta", 0))
                        dep_start = None
                        if delta < 0:
                            if delta == -1 and previous is not None:
                                dep = previous.get(source)
                                dep_start = (
                                    previous_starts.get(source)
                                    if previous_starts is not None else None
                                )
                            elif delta < -1 and panel_history is not None:
                                prior = panel_history.get(panel_index + delta)
                                dep = prior.get(source) if prior is not None else None
                                prior_s = (
                                    panel_start_history.get(panel_index + delta)
                                    if panel_start_history is not None else None
                                )
                                dep_start = (
                                    prior_s.get(source) if prior_s is not None else None
                                )
                            else:
                                dep = None
                        else:
                            dep = group_nodes.get(source) or nodes.get(source)
                            dep_start = group_starts.get(source) or starts.get(source)
                        if dep is not None:
                            deps.append(
                                self._resolve_edge_dep(dep, edge, start_event=dep_start)
                            )
                    elif edge.external_port:
                        external = frame.values.get(edge.external_port)
                        if external is not None:
                            deps.append(self._resolve_edge_dep(external.ready, edge))
                nested_start = None
                nested = getattr(stage, "nested_catalog", None)
                if nested is not None and self._stage_active(stage, frame):
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
                    nested_start = sub.start
                    deps.append(sub.done)
                event = self._submit_catalog_node(
                    stage, frame, name=f"{name}:{prefix}:{stage.label}",
                    deps=tuple(dict.fromkeys(deps)), owner=owner,
                    register=register, ports=ports, allocations=allocations,
                    effects={}, acquired=acquired, port_specs=port_specs,
                    start_event=nested_start, pending_releases=pending_releases,
                    declared_output_ports=ports_by_node.get(local, ()))
                group_nodes[local] = event
                if nested_start is not None:
                    group_starts[local] = nested_start
                    starts[local] = nested_start
                else:
                    starts.pop(local, None)
                created.append(event)
            if pending_out is None and pending_releases:
                self._flush_pending_releases(pending_releases, ports)
            elif pending_out is not None and pending_releases:
                self._flush_pending_releases(pending_releases, ports, only_resolved=True)
            return tuple(created)

        finite = getattr(catalog, "finite", None)
        if finite is None:
            acquired: dict[int, Allocation] = {}
            sinks = {
                int(getattr(stage, "local_id", pos))
                for pos, stage in enumerate(catalog.stages)
            }
            for stage in catalog.stages:
                for edge in getattr(stage, "relative_after", ()):
                    if (edge.source_node is not None
                            and int(getattr(edge, "iteration_delta", 0)) == 0
                            and str(getattr(edge, "source_boundary", "done") or "done") == "done"):
                        sinks.discard(int(edge.source_node))
            stage_events(catalog.stages, binding, "s", (device_root,),
                         acquired=acquired)
            tails = [nodes[local] for local in sorted(sinks) if local in nodes]
            done = self.submit(
                f"{name}:done",
                after=tuple(dict.fromkeys(tails)) or (device_root,),
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

    @staticmethod
    def _lookup_prior_node(prior: Mapping[Any, Event] | None, source: int,
                           class_id: int) -> Event | None:
        if prior is None:
            return None
        dep = prior.get(source)
        if dep is not None:
            return dep
        return prior.get((class_id, source))

    def _resolve_finite_stage_deps(
        self, stage, frame: BindingFrame, *, index: int, class_id: int,
        local: int, roots: tuple[Event, ...], tile_nodes: Mapping[int, Event],
        tile_starts: Mapping[int, Event],
        previous: Mapping[Any, Event],
        previous_starts: Mapping[Any, Event],
        panel_history: Mapping[int, Mapping[Any, Event]],
        panel_start_history: Mapping[int, Mapping[Any, Event]],
    ) -> tuple[Event, ...]:
        stream_sym = getattr(stage, "stream_symbol", None)
        stream_key = int(stream_sym) if stream_sym is not None else (class_id, local)
        deps = list(roots)
        for edge in getattr(stage, "relative_after", ()):
            if edge.source_node is not None:
                source = int(edge.source_node)
                delta = int(getattr(edge, "iteration_delta", 0))
                if delta < 0:
                    prior = (
                        previous if delta == -1
                        else (panel_history.get(index + delta) if index + delta >= 0 else None)
                    )
                    prior_starts = (
                        previous_starts if delta == -1
                        else (panel_start_history.get(index + delta) if index + delta >= 0 else None)
                    )
                    dep = self._lookup_prior_node(prior, source, class_id)
                    dep_start = self._lookup_prior_node(prior_starts, source, class_id)
                else:
                    dep = tile_nodes.get(source)
                    dep_start = tile_starts.get(source)
                if dep is not None:
                    deps.append(self._resolve_edge_dep(dep, edge, start_event=dep_start))
            elif edge.external_port:
                external = frame.values.get(edge.external_port)
                if external is not None:
                    deps.append(self._resolve_edge_dep(external.ready, edge))
        if stream_sym is not None and stream_key in previous:
            deps.append(previous[stream_key])
        return tuple(dict.fromkeys(deps))

    def _submit_finite_catalog(self, catalog, finite, binding, *, name, owner,
                               device_root, cursor, stage_events):
        """[Cohesive-Pipeline] Materialize one bounded live window of a repeated class catalog.

        At most ``finite.slots`` panels carry tasks at any time. Each retired
        panel issues the next absolute index, so the live frontier is
        O(slots * class stages) and independent of ``finite.total``; no Event,
        promise or task exists for an inactive panel. Setup/drain/minimum
        boundaries keep their declared causal position.
        """
        nodes = cursor["nodes"]
        starts = cursor["starts"]
        ports = cursor["ports"]
        allocations = cursor["allocations"]
        port_specs = {
            spec.name: spec for spec in getattr(catalog, "ports", ())
        }
        explicit_out_ports = {
            port
            for st in (*(finite.setup or ()), *(finite.drain or ()),
                       *(s for item in finite.classes for s in item.stages))
            for port in getattr(st, "output_ports", ())
        }
        setup_ids = {int(s.local_id) for s in (finite.setup or ())}
        drain_ids = {int(s.local_id) for s in (finite.drain or ())}
        body_ids = (
            {int(s.local_id) for item in finite.classes for s in item.stages}
            if int(finite.total) > 0 else set()
        )
        setup_out_ports: dict[int, list[str]] = {}
        drain_out_ports: dict[int, list[str]] = {}
        body_first_out_ports: dict[int, list[str]] = {}
        body_last_out_ports: dict[int, list[str]] = {}
        for spec in getattr(catalog, "ports", ()):
            if (getattr(spec, "direction", "in") != "out"
                    or getattr(spec, "producer_node", None) is None
                    or spec.name in explicit_out_ports):
                continue
            pid = int(spec.producer_node)
            if (pid in setup_ids) + (pid in body_ids) + (pid in drain_ids) != 1:
                continue
            if pid in setup_ids:
                setup_out_ports.setdefault(pid, []).append(spec.name)
            elif pid in drain_ids:
                drain_out_ports.setdefault(pid, []).append(spec.name)
            else:
                boundary = str(getattr(spec, "producer_boundary", "done") or "done")
                target_map = body_first_out_ports if boundary == "start" else body_last_out_ports
                target_map.setdefault(pid, []).append(spec.name)
        invocation_acquired: dict[int, Allocation] = {}
        invocation_pending_releases: list = []
        setup_ms = self._scalar(getattr(finite, "setup_ms", 0.0), binding)
        setup_root = (
            self.submit(f"{name}:setup_ms", latency=setup_ms,
                        after=(device_root,), owner=owner)
            if setup_ms > 0.0 else device_root
        )
        setup = tuple(stage_events(
            finite.setup or (), binding, "setup", (setup_root,),
            acquired=invocation_acquired,
            pending_out=invocation_pending_releases,
            declared_ports=setup_out_ports))
        cursor["setup"] = setup
        if not setup:
            post_setup = setup_root
        elif len(setup) == 1:
            post_setup = setup[0]
        else:
            post_setup = self.submit(
                f"{name}:setup_complete", after=setup, owner=owner)
        classes = {
            int(item.class_id): self._order_catalog_stages(item.stages)
            for item in finite.classes
        }
        all_lag_stages = [
            st for ordered in classes.values() for _, st in ordered
        ] + list(finite.drain or ())
        max_lag = max(
            (abs(int(getattr(edge, "iteration_delta", 0)))
             for st in all_lag_stages
             for edge in getattr(st, "relative_after", ())
             if int(getattr(edge, "iteration_delta", 0)) < 0),
            default=1,
        )
        panel_history: dict[int, dict[Any, Event]] = {}
        panel_start_history: dict[int, dict[Any, Event]] = {}
        total = int(finite.total)
        window = self._finite_window(finite)
        minimum = self._scalar(getattr(finite, "minimum_ms", 0.0), binding)
        # Minimum invocation duration anchors after setup completes, matching
        # literal PipelineProgram and PipelineTemplate execution.
        minimum_event = (
            self.submit(f"{name}:minimum", latency=minimum,
                        after=(post_setup,), owner=owner)
            if minimum > 0.0 else None
        )
        done = self.future(f"{name}:done")
        last_tails: list[tuple[Event, ...]] = [(post_setup,)]

        def issue(index: int) -> None:
            raw_class_id, _, _, _, _ = finite.index_of(index)
            class_id = int(raw_class_id)
            ordered = classes.get(class_id)
            if ordered is None:
                raise ExecutionError(f"finite_class_missing:{raw_class_id}")
            frame = self._unit_binding(binding, index, finite)
            roots = (post_setup,)
            acquired = dict(invocation_acquired)
            # Cross-iteration edges resolve against the snapshot taken before
            # this tile is issued; same-tile edges resolve inside ``tile_nodes``.
            previous = dict(cursor["previous"])
            previous_starts = dict(cursor["previous_starts"])
            tile_nodes: dict[int, Event] = {}
            tile_starts: dict[int, Event] = {}
            tile_history: dict[Any, Event] = {}
            tile_start_history: dict[Any, Event] = {}
            pending_releases: list = []
            tails = []
            for local, stage in ordered:
                deps = list(self._resolve_finite_stage_deps(
                    stage, frame, index=index, class_id=class_id,
                    local=local, roots=roots, tile_nodes=tile_nodes,
                    tile_starts=tile_starts, previous=previous,
                    previous_starts=previous_starts,
                    panel_history=panel_history,
                    panel_start_history=panel_start_history))
                nested_start = None
                nested = getattr(stage, "nested_catalog", None)
                if nested is not None and self._stage_active(stage, frame):
                    sub = self.submit_catalog(
                        nested, replace(
                            frame, occurrence=self.new_occurrence(
                                replica=frame.occurrence.replica,
                                rank=frame.occurrence.rank,
                                stage=frame.occurrence.stage)),
                        name=f"{name}:u{index}:{local}:nested", owner=owner,
                        after=tuple(dict.fromkeys(deps)))
                    nested_start = sub.start
                    deps.append(sub.done)
                body_declared = [
                    *(body_first_out_ports.get(local, ()) if index == 0 else ()),
                    *(body_last_out_ports.get(local, ()) if index == total - 1 else ()),
                ]
                event = self._submit_catalog_node(
                    stage, frame, name=f"{name}:u{index}:{stage.label}",
                    deps=tuple(dict.fromkeys(deps)), owner=owner,
                    register=lambda _stage, _event, _local=local:
                        nodes.__setitem__(_local, _event),
                    ports=ports, allocations=allocations, effects={},
                    unit=index, acquired=acquired, port_specs=port_specs,
                    start_event=nested_start, pending_releases=pending_releases,
                    declared_output_ports=body_declared)
                stream_sym = getattr(stage, "stream_symbol", None)
                stream_key = (
                    int(stream_sym) if stream_sym is not None
                    else (class_id, local)
                )
                cursor["previous"][stream_key] = event
                if nested_start is not None:
                    cursor["previous_starts"][stream_key] = nested_start
                    tile_starts[local] = nested_start
                    starts[local] = nested_start
                    if max_lag > 1:
                        tile_start_history[stream_key] = nested_start
                else:
                    cursor["previous_starts"].pop(stream_key, None)
                    starts.pop(local, None)
                if max_lag > 1:
                    tile_history[stream_key] = event
                tile_nodes[local] = event
                nodes[local] = event
                tails.append(event)
            if pending_releases:
                self._flush_pending_releases(pending_releases, ports, only_resolved=True)
                if pending_releases:
                    invocation_pending_releases.extend(pending_releases)
            if invocation_pending_releases:
                self._flush_pending_releases(
                    invocation_pending_releases, ports, only_resolved=True,
                )
            if max_lag > 1:
                panel_history[index] = tile_history
                if tile_start_history:
                    panel_start_history[index] = tile_start_history
                cutoff = index - max(window, max_lag)
                for old_idx in tuple(panel_history):
                    if old_idx < cutoff and old_idx not in cursor["live"]:
                        del panel_history[old_idx]
                        panel_start_history.pop(old_idx, None)
            tile_tails = tuple(tails) or roots
            last_tails[0] = tile_tails
            self.submit(
                f"{name}:u{index}:retire", after=tile_tails, owner=owner,
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
            prev_snap = dict(cursor["previous"])
            prev_starts_snap = dict(cursor["previous_starts"])
            hist_snap = dict(panel_history)
            hist_starts_snap = dict(panel_start_history)
            panel_history.clear()
            panel_start_history.clear()
            tails = []
            drain = tuple(finite.drain or ())
            if drain:
                drain_roots = last_tails[0] if total > 0 else (post_setup,)
                tails.extend(stage_events(
                    drain, binding, "drain", drain_roots,
                    acquired=dict(invocation_acquired),
                    previous=prev_snap, previous_starts=prev_starts_snap,
                    panel_history=hist_snap, panel_start_history=hist_starts_snap,
                    panel_index=total,
                    pending_out=invocation_pending_releases,
                    declared_ports=drain_out_ports))
            elif total <= 0:
                tails.append(post_setup)
            if invocation_pending_releases:
                self._flush_pending_releases(invocation_pending_releases, ports)
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
