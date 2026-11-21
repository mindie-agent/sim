"""Canonical wire contract for immutable rewrite plans.

[Role] Define the eight opcodes, validate their wire representation, and hash plans.
[Invariants] Wire objects are closed, and edits form a dependency DAG.
[Timeouts & Locks] Validation is local and uses no locks or external waits.
[Side-Effect Policy] Serialization and validation are pure.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from functools import cache
from hashlib import sha256
from heapq import heapify, heappop, heappush
import json
from math import isfinite
from types import UnionType
from typing import Any, Literal, TypeAlias, get_args, get_origin, get_type_hints

from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.prediction.streams import (
    MAIN_STREAM, StreamCoreBudget, StreamEvent, StreamId, StreamLane,
)
from mindie_sim.schema.common import ModelingError

REWRITE_PLAN_SCHEMA = "mindie-rewrite-plan/1"
TemplateTier: TypeAlias = Literal["K1", "K2", "K3"]
FusionTier: TypeAlias = Literal["K1_shipping", "K2_epilogue", "K3_dsl_pipeline"]


@dataclass(frozen=True, slots=True)
class OpRef:
    kind: Literal["atom", "result_of"]
    id: str
    output: str = "op"


@dataclass(frozen=True, slots=True)
class PortRef:
    op: OpRef
    name: str
    version: int


@dataclass(frozen=True, slots=True)
class KernelBoundary:
    abi_id: str
    members: tuple[OpRef, ...]


@dataclass(frozen=True, slots=True)
class VariantRef:
    variant_id: str


@dataclass(frozen=True, slots=True)
class FuseRegion:
    members: tuple[OpRef, ...]
    preserve_ports: tuple[PortRef, ...]
    kernel_boundaries: tuple[KernelBoundary, ...]
    fusion_tier: FusionTier = "K1_shipping"


@dataclass(frozen=True, slots=True)
class SelectTemplate:
    target: OpRef
    tier: TemplateTier
    template_id: str
    params: tuple[tuple[str, str | int | bool], ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineTiles:
    target: OpRef
    tile_spec: TilePipelineSpec | VariantRef | None = None
    cohort_split: Literal["none", "dbo_2cohort"] = "none"


@dataclass(frozen=True, slots=True)
class StreamSchedule:
    lanes: tuple[StreamLane, ...]
    assignments: tuple[tuple[OpRef, StreamId], ...] = ()
    events: tuple[StreamEvent, ...] = ()

    def __post_init__(self) -> None:
        ids = {lane.stream_id for lane in self.lanes}
        if not self.lanes or len(ids) != len(self.lanes) or MAIN_STREAM not in ids:
            raise ModelingError("rewrite_stream_lanes_invalid")


@dataclass(frozen=True, slots=True)
class WeightPrefetch:
    target_port: PortRef
    window_anchor: PortRef
    pool: Literal["l1"] = "l1"
    max_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class CollectiveTransform:
    target: OpRef
    mode: str
    topology_route: str


@dataclass(frozen=True, slots=True)
class QuantizeContract:
    scope: tuple[str, ...]
    w_dtype: str
    a_dtype: str
    kv_dtype: str
    group_size: int | None
    scale_dtype: str | None
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not set(self.exclusions) <= set(self.scope):
            raise ModelingError("rewrite_quant_exclusions_outside_scope")


@dataclass(frozen=True, slots=True)
class HoistOrDeduplicate:
    targets: tuple[OpRef, ...]
    scope: Literal["intra_step", "load_time", "device_group_list_elide_d2h"] = "intra_step"


RewriteOp: TypeAlias = (
    FuseRegion | SelectTemplate | PipelineTiles | StreamSchedule | WeightPrefetch
    | CollectiveTransform | QuantizeContract | HoistOrDeduplicate
)
_OPCODES = {opcode.__name__: opcode for opcode in get_args(RewriteOp)}


@dataclass(frozen=True, slots=True)
class Edit:
    edit_id: str
    after: tuple[str, ...] = ()
    operation: RewriteOp = ...  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if type(self.edit_id) is not str or not self.edit_id or self.edit_id in self.after:
            raise ModelingError(f"rewrite_edit_id_or_dependency_invalid:{self.edit_id}")
        if type(self.operation) not in _OPCODES.values():
            raise ModelingError(f"rewrite_edit_operation_invalid:{self.edit_id}")


@dataclass(frozen=True, slots=True)
class RewritePlan:
    schema: Literal["mindie-rewrite-plan/1"] = REWRITE_PLAN_SCHEMA
    base_graph_sha256: str = ""
    framing_sha256: str = ""
    edits: tuple[Edit, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != REWRITE_PLAN_SCHEMA:
            raise ModelingError(f"rewrite_schema_unsupported:{self.schema}")
        _ordered_edits(self.edits)


def _references(value):
    if isinstance(value, OpRef):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _references(item)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _references(getattr(value, field.name))


def _ordered_edits(edits: tuple[Edit, ...], *, handles: bool = False) -> tuple[Edit, ...]:
    by_id = {edit.edit_id: edit for edit in edits}
    if len(by_id) != len(edits):
        raise ModelingError("rewrite_duplicate_edit_id")
    users: dict[str, list[str]] = {key: [] for key in by_id}
    pending: dict[str, int] = {}
    for edit in edits:
        deps = set(edit.after)
        if handles:
            deps.update(ref.id for ref in _references(edit.operation) if ref.kind == "result_of")
        if deps - by_id.keys():
            raise ModelingError(f"rewrite_unknown_edit_dependency:{edit.edit_id}")
        pending[edit.edit_id] = len(deps)
        for dep in deps:
            users[dep].append(edit.edit_id)
    ready = [key for key, count in pending.items() if count == 0]
    heapify(ready)
    result = []
    while ready:
        key = heappop(ready)
        result.append(by_id[key])
        for user in users[key]:
            pending[user] -= 1
            if pending[user] == 0:
                heappush(ready, user)
    if len(result) != len(edits):
        raise ModelingError("rewrite_edit_dependency_cycle")
    return tuple(result)


def _wire(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _wire(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        return {key: _wire(item) for key, item in value.items()}
    if value is None or type(value) in {str, int, bool} or (
        type(value) is float and isfinite(value)
    ):
        return value
    raise ModelingError("rewrite_wire_value_invalid")


def _object(value: Any, allowed: set[str], required: set[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise ModelingError(f"rewrite_wire_object_required:{path}")
    if value.keys() - allowed or not required <= value.keys():
        raise ModelingError(f"rewrite_wire_fields_invalid:{path}")


@cache
def _record_type(cls):
    spec = fields(cls)
    keys = {field.name for field in spec}
    required = {field.name for field in spec
                if field.default is MISSING and field.default_factory is MISSING}
    return keys, required, get_type_hints(cls)


def _record(value: Any, cls: type, path: str):
    keys, required, annotations = _record_type(cls)
    _object(value, keys, required, path)
    return cls(**{
        key: _typed(item, annotations[key], f"{path}.{key}") for key, item in value.items()
    })


def _typed(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is UnionType:
        for option in args:
            try:
                return _typed(value, option, path)
            except ModelingError:
                continue
    elif origin is Literal:
        if any(type(value) is type(option) and value == option for option in args):
            return value
    elif origin is tuple and type(value) is list:
        if args[-1] is Ellipsis:
            return tuple(_typed(item, args[0], path) for item in value)
        if len(value) == len(args):
            return tuple(_typed(item, kind, path) for item, kind in zip(value, args))
    elif origin is Mapping and isinstance(value, Mapping):
        return {_typed(key, args[0], path): _typed(item, args[1], path)
                for key, item in value.items()}
    elif is_dataclass(annotation):
        return _record(value, annotation, path)
    elif type(value) is annotation and (annotation is not float or isfinite(value)):
        return value
    raise ModelingError(f"rewrite_wire_type_invalid:{path}")


def _encode(plan: RewritePlan) -> dict[str, Any]:
    return {
        "schema": plan.schema,
        "base_graph_sha256": plan.base_graph_sha256,
        "framing_sha256": plan.framing_sha256,
        "edits": [
            {"edit_id": edit.edit_id, "after": _wire(edit.after),
             "op": type(edit.operation).__name__, **_wire(edit.operation)}
            for edit in plan.edits
        ],
    }


def encode_rewrite_plan(plan: RewritePlan) -> dict[str, Any]:
    """Encode the plan as a closed, flat-opcode JSON object."""
    payload = _encode(plan)
    decode_rewrite_plan(payload)
    return payload


def decode_rewrite_plan(payload: Mapping[str, Any]) -> RewritePlan:
    """Decode exact JSON types and reject fields outside the current algebra."""
    keys = {"schema", "base_graph_sha256", "framing_sha256", "edits"}
    _object(payload, keys, keys, "plan")
    _typed(payload["schema"], Literal["mindie-rewrite-plan/1"], "schema")
    if type(payload["edits"]) is not list:
        raise ModelingError("rewrite_wire_type_invalid:edits")
    edits = []
    for value in payload["edits"]:
        if not isinstance(value, Mapping) or not {"edit_id", "op"} <= value.keys():
            raise ModelingError("rewrite_wire_edit_fields_invalid")
        opcode = _typed(value["op"], str, "op")
        if opcode not in _OPCODES:
            raise ModelingError(f"rewrite_unknown_opcode:{opcode}")
        operation = _record({key: item for key, item in value.items()
                             if key not in {"edit_id", "after", "op"}}, _OPCODES[opcode], opcode)
        edits.append(Edit(_typed(value["edit_id"], str, "edit_id"),
                          _typed(value.get("after", []), tuple[str, ...], "after"), operation))
    return RewritePlan(base_graph_sha256=_typed(payload["base_graph_sha256"], str, "base_graph_sha256"),
                       framing_sha256=_typed(payload["framing_sha256"], str, "framing_sha256"),
                       edits=tuple(edits))


def _digest_default(value):
    if isinstance(value, PrimitiveStore):
        digest = sha256()
        rows = value.connection.execute(
            "SELECT condition, identity, run_id, repeat_index, duration_ms, source "
            "FROM primitive_samples WHERE split = 'train' ORDER BY condition, run_id, repeat_index")
        for row in rows:
            digest.update(_json_digest(tuple(row)).encode("ascii"))
        return {"training_snapshot_sha256": digest.hexdigest()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"unhashable_scenario_value:{type(value).__name__}")

def _flat(value) -> bool:
    return (value is None or type(value) in {bool, int, str, bytes}
            or (type(value) is float and isfinite(value))
            or (type(value) is tuple and all(type(item) is int for item in value)))


def _json_digest(payload: Any, *, default=None) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False, default=default)
    return sha256(encoded.encode("utf-8")).hexdigest()


def plan_sha256(plan: RewritePlan) -> str:
    """Hash normalized JSON independently of edit and dependency listing order."""
    payload = _encode(plan)
    payload["edits"].sort(key=lambda edit: edit["edit_id"])
    for edit in payload["edits"]:
        edit["after"] = sorted(set(edit["after"]))
    return _json_digest(payload)


def empty_plan(base_graph_sha256: str, framing_sha256: str) -> RewritePlan:
    return RewritePlan(base_graph_sha256=base_graph_sha256, framing_sha256=framing_sha256)


__all__ = [
    "REWRITE_PLAN_SCHEMA", "TemplateTier", "FusionTier", "OpRef", "PortRef",
    "KernelBoundary", "VariantRef", "FuseRegion", "SelectTemplate", "PipelineTiles",
    "StreamSchedule", "WeightPrefetch", "CollectiveTransform", "QuantizeContract",
    "HoistOrDeduplicate", "RewriteOp", "Edit", "RewritePlan", "encode_rewrite_plan",
    "decode_rewrite_plan", "plan_sha256", "empty_plan", "StreamEvent", "StreamId",
    "StreamLane", "StreamCoreBudget", "MAIN_STREAM", "TilePipelineSpec",
]
