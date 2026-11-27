"""Cache execution and occurrence lifecycle.

[Role] Bind requests to cache levels and track live operations and receipts.
[Invariants] Each occurrence uses its own readiness, completion and release events.
[Timeouts & Locks] Live operations and recent receipt history have explicit bounds.
[Side-Effect Policy] Updates cache state and registries and submits work to the Engine.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from mindie_sim.schema.common import ModelingError

from ..cache_path import CacheState
from ..engine import Engine, Event
from .operation import CacheAccessOperation
from .requests import CacheAccessBinding, CacheAccessRequest


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
            "unit": "lines",
            "line_bytes": self.state.geometry.line_bytes,
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
