"""Cache execution requests and typed bindings.

[Role] Define typed producer views, requested accesses and occurrence receipts.
[Invariants] Named source dependencies must refer to matching real producer ports.
[Timeouts & Locks] In-memory validation of per-request declarations.
[Side-Effect Policy] Constructs and validates records without submitting Engine work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mindie_sim.schema.common import ModelingError

from ..cache_path import CacheAccessGeometry, CachePolicy
from ..engine import Event


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
