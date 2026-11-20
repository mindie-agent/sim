"""Event-time execution of a cache access.

[Role] Reserve cache work at readiness and submit its physical stages.
[Invariants] Publication occurs at physical completion. Each occurrence has one
  terminal.
[Timeouts & Locks] Outstanding work is bounded by the reservation and Engine events.
[Side-Effect Policy] Updates the caller-owned reservation and submits work to its
  Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError

from ..cache_path import CacheState, CacheTransfer, cache_work_stages
from ..engine import Allocation, Engine, Event
from .requests import CacheAccessBinding, CacheAccessRequest


@dataclass
class CacheAccessOperation:
    """One executable request bound to one mutable CacheState."""

    state: CacheState
    request: CacheAccessRequest
    hbm_resource: str = "hbm"
    onchip_resource: str = "mte2"
    pool: str | None = "ub"
    write_resource: str | None = None
    vector_resource: str = "vector"

    def __post_init__(self):
        self.started_event: Event | None = None
        self.complete: Event | None = None
        self.ready_event: Event | None = None
        self.staging_allocation: Allocation | None = None
        self.completed_version: int | None = None
        self.launch_name = ""
        self.launch_key = ""
        self.source_event: Event | None = None
        self.stage_events: dict[str, Event] = {}
        # True only after a success/failure terminal has closed this operation.
        self.closed = False

    def _summary(self):
        return self.request.geometry.summary(
            self.state.geometry.line_bytes, write=self.request.write,
            transfer_bytes=self.state.geometry.transfer_bytes)

    def staging_bytes(self) -> int:
        summary = self._summary()
        if not summary.logical_bytes:
            return 0
        return min(summary.gm_bytes,
                   self.state.geometry.line_bytes
                   * self.state.geometry.staging_lines)

    def validate(self, engine: Engine) -> None:
        """Validate every known local resource before the ready task exists."""
        if not isinstance(engine, Engine):
            raise ModelingError("cache_access_operation_requires_engine")
        staging_bytes = self.staging_bytes()
        if staging_bytes and not self.pool:
            raise ModelingError("cache_access_operation_requires_staging_pool")
        if staging_bytes and self.pool not in engine.pools:
            raise ModelingError(f"cache_access_operation_pool_missing:{self.pool}")
        summary = self._summary()
        write_resource = self.write_resource or self.onchip_resource
        unresolved_source = bool(self.request.source_bytes
                                 and self.request.source_residency != "on_chip")
        # Geometry preflight can only know the declared request and an
        # unresolved source. Actual HBM read/write amounts come from the
        # prepared result at ready time.
        possible_hbm = bool(summary.line_count or unresolved_source)
        if possible_hbm and self.hbm_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_operation_hbm_resource_missing:{self.hbm_resource}")
        if summary.logical_bytes:
            for resource in {self.onchip_resource,
                             write_resource if self.request.write else None}:
                if resource and resource not in engine.resources:
                    raise ModelingError(
                        f"cache_access_operation_resource_missing:{resource}")
        elif self.request.source_bytes:
            if self.onchip_resource not in engine.resources:
                raise ModelingError(
                    "cache_access_operation_onchip_resource_missing:"
                    + self.onchip_resource)
        if self.request.convert_ops and self.vector_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_operation_vector_resource_missing:{self.vector_resource}")
        # Credit/response is a declaration on the actual HBM transfer. Check
        # the unit only for a request that can really issue HBM work; a hit
        # request that uses no HBM does not acquire this requirement.
        if ((self.request.outstanding or self.request.response_ms)
                and possible_hbm
                and engine.resources[self.hbm_resource].unit != "bytes"):
            raise ModelingError(
                "cache_access_operation_hbm_resource_not_bytes:"
                + self.hbm_resource)

    def start(self, engine: Engine, name: str, ready: Event, *,
              source: Event | None = None, source_required: bool = False,
              owner: str = "", occurrence: str = "") -> Event:
        if source_required and source is None:
            raise ModelingError("cache_access_operation_source_missing")
        if self.request.source_port and source is None:
            raise ModelingError(
                "cache_access_source_port_unbound:"
                + self.request.source_port)
        self.validate(engine)
        staging_bytes = self.staging_bytes()
        write_resource = self.write_resource or self.onchip_resource
        staging = ()
        if staging_bytes:
            self.staging_allocation = Allocation(
                name + ":staging", self.pool, staging_bytes)
            staging = (self.staging_allocation,)
        complete = engine.future(name + ":complete")
        self.complete = complete
        self.launch_name = name
        # A process-unique occurrence is defined by this operation/Engine pair
        # and this launch, never by a display name or either Engine's local
        # sequence alone.
        self.launch_key = occurrence or (
            f"op{id(self):x}:engine{id(engine):x}#{complete.sequence}")
        self.source_event = source
        line = self.state.geometry.line_bytes

        def close_failed(reservation):
            try:
                if reservation.published:
                    if not reservation.finished:
                        reservation.finish()
                elif not reservation.finished:
                    reservation.cancel()
            except Exception as cleanup:
                reservation.journal.failure += (
                    f";cleanup_failed:{type(cleanup).__name__}:{cleanup}")
                raise
            finally:
                self.closed = True

        def submitted_events():
            return tuple(dict.fromkeys(
                [self.stage_events.get("ready", _ready_task)]
                + [event for label, event in self.stage_events.items()
                   if label != "ready"]))

        work_started = False

        def begin(_):
            nonlocal work_started
            reservation = None
            try:
                reservation = self.state.prepare_access(
                    self.request.identity, self.request.geometry,
                    write=self.request.write, policy=self.request.policy)
                _submit_resolved(reservation)
            except Exception as exc:
                # A failed preparation is never the data-ready completion.
                # Work not yet issued rolls back immediately; already-issued
                # work keeps its physical responsibility until the real
                # terminals complete, then a separate closure releases this
                # reservation and staging without completing a successor.
                if reservation is not None:
                    reservation.fail(exc)
                engine._promises.pop(complete.sequence, None)
                if reservation is None or not work_started:
                    if (reservation is not None and not reservation.published
                            and not reservation.finished):
                        reservation.cancel()
                    if staging:
                        engine.submit(name + ":abort", free=staging,
                                      owner=owner)
                    self.closed = True
                else:
                    try:
                        engine.submit(
                            name + ":failure_close",
                            after=submitted_events(), free=staging,
                            on_complete=lambda _event: close_failed(reservation),
                            owner=owner)
                    except Exception as cleanup:
                        reservation.journal.failure += (
                            ";cleanup_unsubmitted:"
                            f"{type(cleanup).__name__}:{cleanup}")
                        raise ModelingError(
                            "cache_access_failure_cleanup_unsubmitted") from cleanup
                raise

        def _submit_resolved(reservation):
            nonlocal work_started
            result = reservation.result
            events = self.stage_events
            events["ready"] = _ready_task
            items = cache_work_stages(
                result, hbm_resource=self.hbm_resource,
                onchip_resource=self.onchip_resource,
                write_resource=write_resource,
                vector_resource=self.vector_resource,
                source_bytes=self.request.source_bytes,
                source_residency=self.request.source_residency,
                convert_ops=self.request.convert_ops)
            labels = {"ready"}
            for item in items:
                if item.label in labels:
                    raise ModelingError(
                        f"cache_access_operation_duplicate_stage:{item.label}")
                labels.add(item.label)
            for item in items:
                if item.resource and item.resource not in engine.resources:
                    raise ModelingError(
                        "cache_access_operation_resource_missing:"
                        + item.resource)
                if (item.resource == self.hbm_resource
                        and (self.request.outstanding
                             or self.request.response_ms)
                        and engine.resources[item.resource].unit != "bytes"):
                    raise ModelingError(
                        "cache_access_operation_hbm_resource_not_bytes:"
                        + self.hbm_resource)
                missing = tuple(label for label in item.after
                                if label not in labels)
                if missing:
                    raise ModelingError(
                        "cache_access_operation_stage_predecessor_missing:"
                        + ",".join(missing))

            def submit_hbm(item, after):
                amount = item.amount
                if not amount:
                    return after[0]
                label = name + ":" + item.label
                if self.request.outstanding or self.request.response_ms:
                    ready_event = (after[0] if len(after) == 1 else
                                   engine.submit(label + ":join", after=after,
                                                 owner=owner))
                    return CacheTransfer(
                        ((self.hbm_resource, float(amount)),),
                        transaction_bytes=line,
                        outstanding=self.request.outstanding or None,
                        response_ms=self.request.response_ms,
                    ).start(engine, label, ready_event, owner=owner)
                return engine.submit(label, {self.hbm_resource: float(amount)},
                                     after=after, owner=owner)

            def publish_event(_event):
                reservation.publish()

            for item in items:
                after = tuple(events[label] for label in item.after)
                label = name + ":" + item.label
                if item.terminal:
                    def finish(_event):
                        if not reservation.published:
                            reservation.publish()
                        reservation.finish()
                        self.completed_version = self.state.version
                        self.closed = True
                        if not reservation.journal.failure:
                            engine.resolve(complete)
                    event = engine.submit(
                        label, after=after, free=staging,
                        on_complete=finish, owner=owner)
                elif item.publish:
                    work = ({item.resource: item.amount}
                            if item.resource else None)
                    event = engine.submit(
                        label, work, after=after,
                        on_complete=publish_event, owner=owner)
                elif item.label == "drain":
                    if (self.request.outstanding
                            or self.request.response_ms):
                        drained = submit_hbm(item, after)
                    else:
                        drained = engine.submit(
                            label, {self.hbm_resource: float(item.amount)},
                            after=after, owner=owner)
                    event = engine.submit(
                        label + ":terminal", after=(drained,),
                        on_complete=lambda _event: reservation.mark_drained(),
                        owner=owner)
                elif item.resource == self.hbm_resource:
                    event = submit_hbm(item, after)
                else:
                    event = engine.submit(
                        label, {item.resource: item.amount}, after=after,
                        owner=owner)
                self.stage_events[item.label] = event
                reservation.note_issue(item.label, item.resource, item.amount)
                work_started = True

        ready_after = (ready,) if source is None else (ready, source)
        _ready_task = engine.submit(
            name + ":ready", latency=self.request.policy.ready_ms,
            after=ready_after, acquire=staging, on_complete=begin,
            on_start=lambda event: setattr(complete, "started", event.started),
            owner=owner)
        self.ready_event = _ready_task
        self.started_event = _ready_task
        return complete

    def binding(self, occurrence: str | None = None, *, request_id: str = "",
                stage: int = 0, layer: int = 0, rank: int | None = None,
                scope: str = "target", storage: str | None = None, source=None,
                reader=None, grant=None, decision=None,
                last_use=None) -> CacheAccessBinding:
        if self.complete is None:
            raise ModelingError("cache_access_operation_not_started")
        if occurrence is not None and occurrence not in {
                self.launch_name, self.launch_key}:
            raise ModelingError(
                f"cache_access_occurrence_mismatch:{occurrence}")
        if request_id and request_id != self.request.identity:
            raise ModelingError(
                f"cache_access_request_mismatch:{request_id}")
        view = self.request.view
        summary = self._summary()
        actual_source = self.source_event
        if source is not None:
            if actual_source is None:
                # A query cannot create a source for an occurrence that was
                # launched without one.
                raise ModelingError("cache_access_source_not_recorded")
            if source is not actual_source:
                raise ModelingError("cache_access_source_binding_mismatch")
        actual_storage = view.storage if view else ""
        if storage is not None:
            if actual_storage and storage != actual_storage:
                raise ModelingError(
                    f"cache_access_storage_binding_mismatch:{storage}")
            actual_storage = storage
        staging = None
        if self.staging_allocation is not None:
            staging = {
                "name": self.staging_allocation.name,
                "pool": self.staging_allocation.pool,
                "size": self.staging_allocation.size,
            }
        return CacheAccessBinding(
            occurrence=self.launch_key,
            launch_name=self.launch_name,
            request_id=self.request.identity,
            producer=view.producer if view else self.request.producer,
            output=view.output if view else self.request.output,
            output_actual=bool(view and view.output_registered),
            storage=actual_storage,
            dtype=view.dtype if view else self.request.output_dtype,
            layout=view.layout if view else self.request.output_layout,
            owner=view.owner if view else self.request.identity,
            rank=rank,
            state_version=(self.completed_version
                           if self.completed_version is not None
                           else self.state.version),
            source=actual_source,
            source_residency=self.request.source_residency,
            grant=grant,
            staging=staging,
            started=self.ready_event.started if self.ready_event else None,
            completed=self.complete.time if self.complete.done else None,
            event=self.complete,
            last_use=last_use,
            payload_bytes=view.payload_bytes if view else summary.logical_bytes,
            scale_bytes=view.scale_bytes if view else 0,
            padding_bytes=view.padding_bytes if view else 0,
            line_segments=summary.line_segments,
            byte_segments=summary.byte_segments,
            coverage_bytes=summary.coverage_bytes,
            request_token=id(self.request),
        )
