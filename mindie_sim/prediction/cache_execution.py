"""Event-time cache access execution on the shared Engine.

[Role] Bind a typed cache access request to real ready, source, conversion,
  fill, write, staging and completion events. State is resolved at the ready
  event and published at the actual fill/dirty completion event, never at
  graph-build time. Builder and independent execution use this one executor.
[Invariants] A request does not cache a previous result. Same-identity overlap
  and capacity that cannot be reserved are rejected. HBM read/write work,
  source reads, conversion and staging are actually submitted; counters/state
  are published only in the matching completion callbacks. A named dependency
  is a real port: a generic ready cannot substitute for it. Each launch has one
  immutable occurrence receipt; staging is never relabelled as a T08 grant.
  Outstanding transactions are finite.
[Timeouts & Locks] Virtual Engine events only; one reservation per identity.
[Side-Effect Policy] Mutates the caller's CacheState and Engine only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .cache_path import (
    CacheAccessGeometry, CachePolicy, CacheState, CacheTransfer,
    cache_work_stages,
)
from .engine import Allocation, Engine, Event


@dataclass(frozen=True)
class CacheTypedView:
    """Actual typed producer/output/view carried by one cache request."""

    producer: str
    output: str
    dtype: str
    layout: str
    shape: tuple[int, int]
    storage: str
    owner: str = ""
    inputs: tuple[str, ...] = ()
    payload_bytes: int = 0
    scale_bytes: int = 0
    packed_elements: int = 1
    group_size: int = 0
    #: True only when the producer's real output namespace registered this
    #: name. A view without one is a declaration, not an actual port.
    output_registered: bool = False
    padding_bytes: int = 0

    def __post_init__(self):
        for name in ("producer", "output", "dtype", "layout", "storage", "owner"):
            if not isinstance(getattr(self, name), str):
                raise ModelingError(f"cache_typed_view_{name}_must_be_string")
        if not self.producer or not self.output:
            raise ModelingError("cache_typed_view_requires_producer_and_output")
        if (not isinstance(self.shape, tuple) or len(self.shape) != 2
                or any(type(item) is not int or item < 0 for item in self.shape)):
            raise ModelingError("cache_typed_view_shape_must_be_two_integers")
        if type(self.payload_bytes) is not int or self.payload_bytes < 0:
            raise ModelingError("cache_typed_view_payload_bytes_invalid")
        if type(self.scale_bytes) is not int or self.scale_bytes < 0:
            raise ModelingError("cache_typed_view_scale_bytes_invalid")
        if type(self.padding_bytes) is not int or self.padding_bytes < 0:
            raise ModelingError("cache_typed_view_padding_bytes_invalid")
        if type(self.packed_elements) is not int or self.packed_elements < 1:
            raise ModelingError("cache_typed_view_packed_elements_invalid")
        if type(self.group_size) is not int or self.group_size < 0:
            raise ModelingError("cache_typed_view_group_size_invalid")
        if type(self.output_registered) is not bool:
            raise ModelingError("cache_typed_view_output_registered_invalid")
        if not isinstance(self.inputs, tuple) or any(
                not isinstance(item, str) for item in self.inputs):
            raise ModelingError("cache_typed_view_inputs_must_be_strings")

    def to_dict(self) -> dict:
        return {
            "producer": self.producer, "output": self.output,
            "dtype": self.dtype, "layout": self.layout,
            "shape": list(self.shape), "storage": self.storage,
            "owner": self.owner, "inputs": list(self.inputs),
            "payload_bytes": self.payload_bytes,
            "scale_bytes": self.scale_bytes,
            "padding_bytes": self.padding_bytes,
            "packed_elements": self.packed_elements,
            "group_size": self.group_size,
            "output_registered": self.output_registered,
        }


@dataclass(frozen=True)
class CacheAccessRequest:
    """Immutable request; safe to reuse across serving iterations."""

    identity: str
    geometry: CacheAccessGeometry
    write: bool = False
    policy: CachePolicy = field(default_factory=CachePolicy)
    outstanding: int = 0
    response_ms: float = 0.0
    source_bytes: int = 0
    convert_ops: float = 0.0
    source_residency: str = "unresolved"
    deps: tuple = ()
    producer: str = ""
    output: str = ""
    output_dtype: str = ""
    output_layout: str = ""
    view: CacheTypedView | None = None
    source_port: str = ""

    def __post_init__(self):
        if not isinstance(self.identity, str) or not self.identity:
            raise ModelingError("cache_access_request_identity_must_be_nonempty")
        if not isinstance(self.geometry, CacheAccessGeometry):
            raise ModelingError("cache_access_request_requires_typed_geometry")
        if type(self.write) is not bool:
            raise ModelingError("cache_access_request_write_must_be_boolean")
        if not isinstance(self.policy, CachePolicy):
            raise ModelingError("cache_access_request_requires_cache_policy")
        if type(self.outstanding) is not int or self.outstanding < 0:
            raise ModelingError("cache_access_request_outstanding_must_be_nonnegative")
        if (not isinstance(self.response_ms, (int, float))
                or isinstance(self.response_ms, bool) or not self.response_ms >= 0):
            raise ModelingError("cache_access_request_response_must_be_nonnegative")
        if type(self.source_bytes) is not int or self.source_bytes < 0:
            raise ModelingError("cache_access_request_source_bytes_must_be_nonnegative")
        if (not isinstance(self.convert_ops, (int, float))
                or isinstance(self.convert_ops, bool) or self.convert_ops < 0):
            raise ModelingError("cache_access_request_convert_ops_must_be_nonnegative")
        if self.source_residency not in {"unresolved", "on_chip"}:
            raise ModelingError("cache_access_request_unknown_source_residency")
        if not isinstance(self.deps, tuple):
            raise ModelingError("cache_access_request_deps_must_be_tuple")
        if any(not isinstance(dep, str) or not dep for dep in self.deps):
            raise ModelingError("cache_access_request_deps_must_be_nonempty_strings")
        if len(set(self.deps)) != len(self.deps):
            raise ModelingError("cache_access_request_deps_must_be_unique")
        if not isinstance(self.producer, str) or not isinstance(self.output, str):
            raise ModelingError("cache_access_request_producer_output_must_be_strings")
        if not isinstance(self.output_dtype, str) or not isinstance(
                self.output_layout, str):
            raise ModelingError("cache_access_request_output_type_must_be_strings")
        if self.view is not None and not isinstance(self.view, CacheTypedView):
            raise ModelingError("cache_access_request_view_must_be_typed_view")
        if not isinstance(self.source_port, str):
            raise ModelingError("cache_access_request_source_port_must_be_string")

    @property
    def logical_bytes(self) -> int:
        return self.geometry.logical_bytes

    @property
    def effective_producer(self) -> str:
        if self.view is not None:
            return self.view.producer
        return self.producer

    @property
    def effective_output(self) -> str:
        if self.view is not None:
            return self.view.output
        return self.output


@dataclass(frozen=True)
class CacheAccessBinding:
    """One immutable actual occurrence receipt.

    ``occurrence`` is the unique launch key issued at submission. ``source`` is
    the event that was actually passed to ``start``. ``grant`` stays ``None``
    until a real T08 backing grant is supplied; the Engine staging allocation is
    reported separately as ``staging`` and is never relabelled as that grant.
    """

    occurrence: str
    launch_name: str
    request_id: str
    producer: str
    output: str
    output_actual: bool
    storage: str
    dtype: str
    layout: str
    owner: str
    rank: int | None
    state_version: object
    source: object
    source_residency: str
    grant: object
    staging: dict | None
    started: float | None
    completed: float | None
    event: Event
    last_use: object
    payload_bytes: int
    scale_bytes: int
    line_segments: tuple
    byte_segments: tuple
    coverage_bytes: int
    #: Exact request object token; a same-identity but different request is
    #: not allowed to dereference this receipt.
    request_token: int = 0
    padding_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "occurrence": self.occurrence,
            "launch_name": self.launch_name,
            "request": self.request_id,
            "producer": self.producer, "output": self.output,
            "output_actual": self.output_actual,
            "storage": self.storage, "dtype": self.dtype,
            "layout": self.layout, "owner": self.owner,
            "rank": self.rank, "state_version": self.state_version,
            "source": self.source, "source_residency": self.source_residency,
            "grant": self.grant, "staging": self.staging,
            "started": self.started,
            "completed": self.completed, "last_use": self.last_use,
            "payload_bytes": self.payload_bytes,
            "scale_bytes": self.scale_bytes,
            "padding_bytes": self.padding_bytes,
            "line_segments": [item.to_tuple() for item in self.line_segments],
            "byte_segments": [item.to_tuple() for item in self.byte_segments],
            "coverage_bytes": self.coverage_bytes,
        }


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


@dataclass
class CacheExecution:
    """Same-context cache execution boundary reusable across iterations."""

    state: CacheState
    hbm_resource: str = "hbm"
    onchip_resource: str = "mte2"
    pool: str | None = "ub"
    vector_resource: str = "vector"
    write_resource: str = "mte3"

    _RECEIPT_LIMIT = 256

    def __post_init__(self):
        # Unique launch key -> operation. Labels are indexes, never the
        # identity of a receipt; a held receipt therefore outlives re-execution.
        self._operations: dict[str, CacheAccessOperation] = {}
        self._labels: dict[str, list[str]] = {}
        self._order: deque[str] = deque()
        self._occurrence_sequence = 0

    def _adopt_runtime(self, engine: Engine) -> None:
        """Claim the one runtime cache domain when that root is available.

        The new causal Engine exposes ``cache_domains``; the still-public legacy
        Engine object in this parallel construction window does not.  No
        execution path is selected here: cache math is identical, only the
        owner of the mutable residency root is recorded when it exists.
        """
        if getattr(engine, "cache_domains", None) is not None:
            if (self.state.runtime_uid is not None
                    and self.state.runtime_uid != getattr(engine, "uid", None)
                    and not self.state._inflight
                    and self.state._pending_writeback is None
                    and all(op.closed for op in self._operations.values())):
                self.state.runtime_uid = None
            self.state.bind_runtime(engine)

    def _register(self, operation: CacheAccessOperation) -> None:
        key = operation.launch_key
        if not key or key in self._operations:
            raise ModelingError(f"cache_access_occurrence_not_unique:{key}")
        self._operations[key] = operation
        self._labels.setdefault(operation.launch_name, []).append(key)
        self._order.append(key)
        # Reclaim only retired receipts. An in-flight operation keeps its
        # receipt regardless of how many finished accesses preceded it.
        while len(self._order) > self._RECEIPT_LIMIT:
            expired = None
            for candidate in self._order:
                operation_at = self._operations.get(candidate)
                if operation_at is not None and operation_at.closed:
                    expired = candidate
                    break
            if expired is None:
                break
            self._order.remove(expired)
            self._operations.pop(expired, None)
            for label, keys in tuple(self._labels.items()):
                if expired in keys:
                    keys.remove(expired)
                    if not keys:
                        self._labels.pop(label, None)
                    break

    def _find(self, occurrence: str,
              request: CacheAccessRequest | None = None
              ) -> CacheAccessOperation | None:
        operation = self._operations.get(occurrence)
        if operation is not None:
            if request is not None and operation.request is not request:
                return None
            return operation
        candidates = [self._operations[key]
                      for key in self._labels.get(occurrence, ())
                      if key in self._operations]
        if request is not None:
            candidates = [item for item in candidates
                          if item.request is request]
        return candidates[-1] if candidates else None

    def _request_known(self, request: CacheAccessRequest) -> bool:
        return any(operation.request is request
                   for operation in self._operations.values())

    def _label_known(self, occurrence: str) -> bool:
        return occurrence in self._labels or occurrence in self._operations

    def release(self, occurrence: str) -> bool:
        """Release one lookup index; a caller-held receipt stays unchanged."""
        operation = self._find(occurrence)
        if operation is None:
            return False
        key = operation.launch_key
        self._operations.pop(key, None)
        keys = self._labels.get(operation.launch_name, [])
        if key in keys:
            keys.remove(key)
            if not keys:
                self._labels.pop(operation.launch_name, None)
        try:
            self._order.remove(key)
        except ValueError:
            pass
        return True

    def access(self, request: CacheAccessRequest, engine: Engine, name: str,
               ready: Event, *, source: Event | None = None,
               source_required: bool = False, owner: str = "",
               deps: Mapping[str, Event] | None = None,
               ports_verified: bool = False) -> Event:
        """Preflight declared ports, then submit exactly one occurrence.

        ``deps`` is the verified event map for ``request.deps``. A direct call
        with named deps must supply it; ``CacheLowering.execute`` supplies the
        fully preflighted map and sets ``ports_verified``. A generic ready is
        never substituted for a named source.
        """
        if deps is not None:
            if not isinstance(deps, Mapping):
                raise ModelingError("cache_access_deps_must_be_mapping")
            for key, value in deps.items():
                if not isinstance(key, str) or not key:
                    raise ModelingError(
                        "cache_access_dep_name_must_be_nonempty_string")
                if not isinstance(value, Event):
                    raise ModelingError(
                        f"cache_access_dep_event_invalid:{key}")
        if request.deps:
            # A verified caller still has to hand over the real event map:
            # the flag proves preflight, it does not replace the ports.
            if deps is None:
                raise ModelingError(
                    "cache_access_dependency_unbound:"
                    + ",".join(request.deps))
            missing = tuple(dep for dep in request.deps if dep not in deps)
            if missing:
                raise ModelingError(
                    "cache_access_dependency_unbound:" + ",".join(missing))
        source_event = source
        if request.source_port:
            if source_event is None and deps is not None:
                source_event = deps.get(request.source_port)
            if source_event is None:
                raise ModelingError(
                    "cache_access_source_port_unbound:"
                    + request.source_port)
        if source_required and source_event is None:
            raise ModelingError("cache_access_operation_source_missing")
        self._adopt_runtime(engine)
        operation = CacheAccessOperation(
            self.state, request, self.hbm_resource, self.onchip_resource,
            self.pool, self.write_resource, self.vector_resource)
        operation.validate(engine)
        self._occurrence_sequence += 1
        occurrence = (f"exec{id(self):x}:engine{id(engine):x}:"
                      f"{self._occurrence_sequence}")
        operation.launch_name = name
        operation.launch_key = occurrence
        # Duplicate registration is rejected before any Engine mutation.
        self._register(operation)
        try:
            # Even a verified plan still joins its real dependencies. The
            # caller's preflight may already have merged them, but the join
            # event is the causal boundary this operation actually observes.
            predecessors = [ready]
            if deps is not None:
                predecessors.extend(deps[dep] for dep in request.deps)
            unique = tuple({event.sequence: event for event in predecessors}
                           .values())
            if len(unique) > 1 or (deps and request.deps):
                ready = engine.submit(name + ":ports", after=unique,
                                      owner=owner)
            event = operation.start(
                engine, name, ready, source=source_event,
                source_required=source_required, owner=owner,
                occurrence=occurrence)
        except Exception:
            self.release(occurrence)
            raise
        return event

    def binding(self, occurrence: str) -> CacheAccessBinding | None:
        operation = self._find(occurrence)
        if operation is None:
            return None
        return operation.binding(occurrence)

    def binding_of(self, request: CacheAccessRequest, occurrence: str, *,
                   storage: str | None = None) -> CacheAccessBinding | None:
        operation = self._find(occurrence, request)
        if operation is None:
            return None
        return operation.binding(occurrence, storage=storage)

    def state_access(self, request: CacheAccessRequest, *, occurrence: str,
                     storage: str | None = None, request_id: str = "",
                     stage: int = 0, layer: int = 0, rank: int | None = None,
                     scope: str = "target", state_version=None, source=None,
                     reader=None, grant=None, decision=None,
                     last_use=None, binding: CacheAccessBinding | None = None
                     ) -> dict:
        """Project a request onto the existing SERVICE state-access fields.

        A runtime binding dereferences the actual immutable occurrence receipt.
        Without one, the result is explicitly marked as a declaration; a ghost
        occurrence or a mismatched caller-held receipt is rejected.
        """
        summary = request.geometry.summary(
            self.state.geometry.line_bytes, write=request.write,
            transfer_bytes=self.state.geometry.transfer_bytes)
        if binding is None:
            operation = self._find(occurrence, request)
            if operation is not None:
                binding = operation.binding(
                    occurrence, storage=storage, source=source,
                    grant=grant, last_use=last_use)
            elif self._request_known(request) or self._label_known(occurrence):
                raise ModelingError(
                    f"cache_access_occurrence_unknown:{occurrence}")
        else:
            if binding.request_id != request.identity:
                raise ModelingError(
                    f"cache_access_request_mismatch:{binding.request_id}")
            if binding.request_token and binding.request_token != id(request):
                raise ModelingError("cache_access_request_mismatch:receipt")
            if occurrence not in {binding.occurrence, binding.launch_name}:
                raise ModelingError(
                    f"cache_access_binding_mismatch:{occurrence}")
            if (binding.storage and storage is not None
                    and storage != binding.storage):
                raise ModelingError(
                    f"cache_access_storage_binding_mismatch:{storage}")
            if source is not None:
                if binding.source is None:
                    raise ModelingError("cache_access_source_not_recorded")
                if source is not binding.source:
                    raise ModelingError("cache_access_source_binding_mismatch")
            if (binding.grant is not None and grant is not None
                    and grant is not binding.grant):
                raise ModelingError("cache_access_grant_binding_mismatch")
            if (state_version is not None
                    and binding.state_version != state_version):
                raise ModelingError("cache_access_version_binding_mismatch")
        view = request.view
        producer = (binding.producer if binding else
                    (view.producer if view else request.producer))
        output = (binding.output if binding else
                  (view.output if view else request.output))
        actual_storage = (binding.storage if binding else
                          ((view.storage if view else "") or (storage or "")))
        dtype = binding.dtype if binding else (
            view.dtype if view else request.output_dtype)
        layout = binding.layout if binding else (
            view.layout if view else request.output_layout)
        owner = binding.owner if binding else (
            view.owner if view else request.identity)
        version = (binding.state_version if binding
                   else (state_version if state_version is not None
                         else self.state.version))
        completion = binding.completed if binding else None
        started = binding.started if binding else None
        grant_value = (binding.grant if binding and binding.grant is not None
                       else grant)
        source_value = binding.source if binding else source
        last_use_value = (binding.last_use
                          if binding and binding.last_use is not None
                          else last_use)
        output_actual = bool(binding and binding.output_actual)
        staging = binding.staging if binding else None
        gaps = []
        if binding is None:
            gaps.append("runtime_binding_not_established")
        if not output_actual:
            gaps.append("typed_output_not_registered")
        if grant_value is None:
            gaps.append("T08_grant_not_connected")
        if last_use_value is None:
            gaps.append("T08_last_use_not_connected")
        actual_occurrence = binding.occurrence if binding else occurrence
        launch_name = binding.launch_name if binding else ""
        request_value = (binding.request_id if binding
                         else (request_id or request.identity))
        line_segments = summary.line_segments
        byte_segments = summary.byte_segments
        if summary.ranges_complete:
            span = [[start, count] for start, count in summary.line_ranges]
            byte_span = [[start, count] for start, count in summary.byte_ranges]
        else:
            span = [list(item.to_tuple()) for item in line_segments]
            byte_span = [list(item.to_tuple()) for item in byte_segments]
        return {
            "occurrence": actual_occurrence, "launch_name": launch_name,
            "producer": producer,
            "request": request_value, "stage": stage, "layer": layer,
            "direction": "write" if request.write else "read",
            "position": "cache", "output": output,
            "output_actual": output_actual,
            "storage": actual_storage, "dtype": dtype,
            "layout": layout, "owner": owner,
            "state_version": version,
            "span": span,
            "byte_span": byte_span,
            "span_format": ("start_count" if summary.ranges_complete
                            else "start_width_stride_count"),
            "line_segments": [list(item.to_tuple()) for item in line_segments],
            "byte_segments": [list(item.to_tuple()) for item in byte_segments],
            "coverage_bytes": summary.coverage_bytes,
            "ranges_complete": summary.ranges_complete,
            "payload_bytes": (binding.payload_bytes if binding
                              else (view.payload_bytes if view
                                    else summary.logical_bytes)),
            "scale_bytes": (binding.scale_bytes if binding
                            else (view.scale_bytes if view else 0)),
            "padding_bytes": (binding.padding_bytes if binding
                              else (view.padding_bytes if view else 0)),
            "rank": rank, "scope": scope,
            "bytes": summary.logical_bytes,
            "source": source_value, "source_actual": bool(
                binding and binding.source is not None),
            "reader": reader, "grant": grant_value,
            "grant_connected": grant_value is not None,
            "staging": staging,
            "started": started, "completed": completion,
            "decision": decision, "last_use": last_use_value,
            "last_use_connected": last_use_value is not None,
            "runtime_bound": binding is not None,
            "evidence": "runtime" if binding is not None else "declaration",
            "gaps": gaps,
        }

    def flush(self, engine: Engine, name: str, ready: Event, *,
              owner: str = "") -> Event:
        """Resolve the dirty version only when the supplied ready event fires."""
        self._adopt_runtime(engine)
        complete = engine.future(name + ":complete")

        def begin(_):
            plan = self.state.flush()
            done = plan.start(engine, name, _ready_task,
                              hbm_resource=self.hbm_resource, owner=owner)
            engine.submit(name + ":join", after=(done,),
                          on_complete=lambda _: engine.resolve(complete),
                          owner=owner)

        _ready_task = engine.submit(name + ":ready", after=(ready,),
                                    on_complete=begin, owner=owner)
        return complete


__all__ = [
    "CacheAccessBinding", "CacheAccessOperation", "CacheAccessRequest",
    "CacheExecution", "CacheTypedView",
]
