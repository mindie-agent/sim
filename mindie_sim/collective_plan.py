"""Explicit, non-streaming collective rendezvous on the shared resource engine.

Every participant contributes a producer dependency, including zero-payload
participants.  Physical link work is declared once for the whole collective;
logical payload is evidence and is never multiplied into another traffic bill.
This module does not infer a transport algorithm, endpoint work, or EP/DCP groups.
No network, device, filesystem or global mutation is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .a3_resource_engine import ResourceEngine, ResourceSpec
from .a3_types import A3InputError

_KINDS = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "barrier"}
_KEY_FIELDS = {"kind", "group", "epoch", "stage", "layer"}


def _text(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A3InputError(reason)
    return value


def _integer(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise A3InputError(reason)
    return value


def _number(value: Any, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A3InputError(reason)
    try:
        result = float(value)
    except OverflowError as exc:
        raise A3InputError(reason) from exc
    if not math.isfinite(result) or result < 0:
        raise A3InputError(reason)
    return result


def _sequence(value: Any, reason: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise A3InputError(reason)
    return value


def _mapping(value: Any, allowed: set[str], reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise A3InputError(reason)
    return value


def _rank_map(value: Any, reason: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise A3InputError(reason)
    result: dict[int, Any] = {}
    for key, item in value.items():
        # JSON has string keys; reject noncanonical spellings and bool ranks.
        if isinstance(key, str) and key.isascii() and key.isdigit():
            rank = int(key)
            if str(rank) != key:
                raise A3InputError(reason)
        else:
            rank = _integer(key, reason)
        if rank in result:
            raise A3InputError(reason)
        result[rank] = item
    return result


def _names(value: Any, reason: str, *, nonempty: bool = False) -> tuple[str, ...]:
    items = tuple(_text(item, reason) for item in _sequence(value, reason))
    if len(set(items)) != len(items) or (nonempty and not items):
        raise A3InputError(reason)
    return items


def _work(value: Any, reason: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        raise A3InputError(reason)
    return tuple(sorted(
        (_text(name, reason), _number(amount, reason))
        for name, amount in value.items()
    ))


@dataclass(frozen=True)
class CollectiveKey:
    """Communicator operation identity; no implicit world or epoch matching."""

    kind: str
    group: str
    epoch: int
    stage: int
    layer: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise A3InputError("collective_kind_unsupported")
        _text(self.group, "collective_group_invalid")
        for name in ("epoch", "stage", "layer"):
            _integer(getattr(self, name), f"collective_{name}_invalid")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CollectiveKey:
        data = _mapping(raw, _KEY_FIELDS, "collective_identity_invalid")
        if set(data) != _KEY_FIELDS:
            raise A3InputError("collective_identity_incomplete")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(_KEY_FIELDS)}

    @property
    def task_prefix(self) -> str:
        # Hash the structured key instead of joining user strings with a delimiter.
        token = hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()
        return f"collective:{token}"


@dataclass(frozen=True)
class CollectiveLinkStage:
    """Physical wire-byte work for one explicit stage, not logical payload."""

    name: str
    link_bytes: tuple[tuple[str, float], ...]
    deps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "collective_link_stage_name_invalid")
        names: set[str] = set()
        for name, amount in self.link_bytes:
            if not isinstance(name, str) or not name.startswith("link:") or name in names:
                raise A3InputError("collective_link_work_invalid")
            _number(amount, "collective_link_work_invalid")
            names.add(name)
        _names(self.deps, "collective_link_stage_dependencies_invalid")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CollectiveLinkStage:
        data = _mapping(raw, {"name", "link_bytes", "deps"}, "collective_link_stage_invalid")
        return cls(
            name=data.get("name"),
            link_bytes=_work(data.get("link_bytes", {}), "collective_link_work_invalid"),
            deps=_names(data.get("deps", ()), "collective_link_stage_dependencies_invalid"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "link_bytes": dict(self.link_bytes), "deps": list(self.deps)}


@dataclass(frozen=True)
class CollectiveDescriptor:
    """One non-streaming operation with an exact participant/producer contract."""

    key: CollectiveKey
    participants: tuple[int, ...]
    payload_bytes_by_rank: tuple[tuple[int, float], ...]
    producers_by_rank: tuple[tuple[int, tuple[str, ...]], ...]
    link_stages: tuple[CollectiveLinkStage, ...]
    source: str = "declared"

    def __post_init__(self) -> None:
        if not isinstance(self.key, CollectiveKey):
            raise A3InputError("collective_identity_invalid")
        ranks = tuple(_integer(item, "collective_participant_invalid") for item in self.participants)
        if not ranks or len(set(ranks)) != len(ranks):
            raise A3InputError("collective_participants_empty_or_duplicate")
        expected = set(ranks)
        for field, rows in (("payload", self.payload_bytes_by_rank), ("producer", self.producers_by_rank)):
            declared = [_integer(rank, "collective_member_invalid") for rank, _ in rows]
            if len(set(declared)) != len(declared):
                raise A3InputError(f"collective_{field}_member_duplicate")
            if set(declared) != expected:
                missing = sorted(expected - set(declared))
                extra = sorted(set(declared) - expected)
                raise A3InputError(f"collective_{field}_members_mismatch:missing={missing}:extra={extra}")
        for _, amount in self.payload_bytes_by_rank:
            _number(amount, "collective_payload_invalid")
        for _, tasks in self.producers_by_rank:
            _names(tasks, "collective_producers_empty_or_duplicate", nonempty=True)
        _text(self.source, "collective_source_invalid")
        seen: set[str] = set()
        for stage in self.link_stages:
            if not isinstance(stage, CollectiveLinkStage) or stage.name in seen:
                raise A3InputError("collective_link_stage_duplicate_or_invalid")
            if any(dep not in seen for dep in stage.deps):
                raise A3InputError(f"collective_link_stage_dependency_not_prior:{stage.name}")
            seen.add(stage.name)
        if not self.link_stages and self.key.kind != "barrier":
            raise A3InputError("collective_link_stages_required")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CollectiveDescriptor:
        data = _mapping(raw, _KEY_FIELDS | {
            "participants", "payload_bytes_by_rank", "producers_by_rank", "link_stages", "source",
        }, "collective_descriptor_invalid")
        return cls(
            key=CollectiveKey.from_mapping({key: data[key] for key in _KEY_FIELDS if key in data}),
            participants=tuple(_sequence(data.get("participants"), "collective_participants_invalid")),
            payload_bytes_by_rank=tuple(sorted(_rank_map(
                data.get("payload_bytes_by_rank"), "collective_payload_members_invalid",
            ).items())),
            producers_by_rank=tuple(sorted(
                (rank, _names(tasks, "collective_producers_empty_or_duplicate", nonempty=True))
                for rank, tasks in _rank_map(
                    data.get("producers_by_rank"), "collective_producer_members_invalid",
                ).items()
            )),
            link_stages=tuple(CollectiveLinkStage.from_mapping(item) for item in _sequence(
                data.get("link_stages", ()), "collective_link_stages_invalid",
            )),
            source=data.get("source", "declared"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.key.to_dict(), "participants": list(self.participants),
            "payload_bytes_by_rank": {str(rank): value for rank, value in self.payload_bytes_by_rank},
            "producers_by_rank": {str(rank): list(tasks) for rank, tasks in self.producers_by_rank},
            "link_stages": [stage.to_dict() for stage in self.link_stages], "source": self.source,
        }


@dataclass(frozen=True)
class CollectiveRegistration:
    """Returned gates can be used as dependencies of later engine tasks."""

    descriptor: CollectiveDescriptor
    join_task_id: str
    stage_task_ids: tuple[tuple[str, str], ...]
    completion_task_id: str
    consumer_gates: tuple[tuple[int, str], ...]

    def consumer_gate(self, rank: int) -> str:
        _integer(rank, "collective_consumer_rank_invalid")
        if rank not in dict(self.consumer_gates):
            raise A3InputError(f"collective_consumer_not_participant:{rank}")
        return dict(self.consumer_gates)[rank]


def register_collective(
    engine: ResourceEngine, descriptor: CollectiveDescriptor,
) -> CollectiveRegistration:
    """Append join, one wire-work DAG and zero-work participant gates.

    Register every producer first, call this function, then register consumers
    with ``dependencies=(registration.consumer_gate(rank),)``.  All validation
    is completed before mutating the engine.  Existing physical link resources
    are reused; no shadow wire, endpoint work or global barrier is introduced.
    """
    if not isinstance(descriptor, CollectiveDescriptor):
        raise A3InputError("collective_descriptor_invalid")
    prefix = descriptor.key.task_prefix
    join_id, done_id = f"{prefix}:join", f"{prefix}:complete"
    stage_ids = tuple((stage.name, f"{prefix}:wire:{index}")
                      for index, stage in enumerate(descriptor.link_stages))
    gates = tuple((rank, f"{prefix}:rank:{rank}:ready") for rank in descriptor.participants)
    generated = (join_id, done_id, *(task for _, task in stage_ids), *(task for _, task in gates))
    if any(task in engine.tasks or task in engine.completed_at for task in generated):
        raise A3InputError("collective_identity_or_task_collision")
    producers = tuple(dict.fromkeys(
        task for _, tasks in descriptor.producers_by_rank for task in tasks
    ))
    unknown = [task for task in producers if task not in engine.tasks and task not in engine.completed_at]
    if unknown:
        raise A3InputError("collective_producer_unknown:" + ",".join(unknown))
    for stage in descriptor.link_stages:
        for name, _ in stage.link_bytes:
            resource = engine.resources.get(name)
            if resource is None or resource.unit != "bytes":
                raise A3InputError(f"collective_physical_link_missing_or_unit_invalid:{name}")
    identity = descriptor.key.to_dict()
    engine.add_task(join_id, {}, dependencies=producers, owner="collective_join", metadata=identity)
    stage_map = dict(stage_ids)
    for stage in descriptor.link_stages:
        engine.add_task(
            stage_map[stage.name], dict(stage.link_bytes),
            dependencies=tuple(stage_map[dep] for dep in stage.deps) if stage.deps else (join_id,),
            owner="collective_wire", metadata={**identity, "link_stage": stage.name},
        )
    depended_on = {dep for stage in descriptor.link_stages for dep in stage.deps}
    leaves = tuple(task for name, task in stage_ids if name not in depended_on)
    engine.add_task(done_id, {}, dependencies=leaves or (join_id,), owner="collective_complete", metadata=identity)
    for rank, task in gates:
        engine.add_task(task, {}, dependencies=(done_id,), owner="collective_consumer_gate",
                        metadata={**identity, "rank": rank})
    return CollectiveRegistration(descriptor, join_id, stage_ids, done_id, gates)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _register_work(engine: ResourceEngine, raw: Mapping[str, Any], *, deps: Sequence[str] = ()) -> None:
    data = _mapping(raw, {"task_id", "demands", "dependencies", "release_ms"}, "collective_work_task_invalid")
    task_id = _text(data.get("task_id"), "collective_work_task_id_invalid")
    engine.add_task(
        task_id, dict(_work(data.get("demands", {}), "collective_work_demands_invalid")),
        dependencies=tuple(dict.fromkeys((*_names(
            data.get("dependencies", ()), "collective_work_dependencies_invalid",
        ), *deps))),
        release_ms=_number(data.get("release_ms", 0), "collective_work_release_invalid"),
        owner="declared_work",
    )


def _report_collective(registration: CollectiveRegistration, engine: ResourceEngine) -> dict[str, Any]:
    declared: dict[str, float] = {}
    serviced: dict[str, float] = {}
    for _, task_id in registration.stage_task_ids:
        task = engine.tasks[task_id]
        for name, amount in task.demands.items():
            declared[name] = declared.get(name, 0.0) + amount
        for name, start, end, rate in task.service_intervals:
            serviced[name] = serviced.get(name, 0.0) + (end - start) * rate
    conserved = all(math.isclose(amount, serviced.get(name, 0.0), rel_tol=1e-10, abs_tol=0.0)
                    for name, amount in declared.items())
    if not conserved:
        raise A3InputError("collective_link_work_not_conserved")
    descriptor = registration.descriptor
    return {
        "descriptor": descriptor.to_dict(),
        "task_ids": {
            "join": registration.join_task_id, "wire_stages": dict(registration.stage_task_ids),
            "complete": registration.completion_task_id,
            "consumer_gates": {str(rank): task for rank, task in registration.consumer_gates},
        },
        "producer_completion_ms": {
            str(rank): {task: engine.completed_at[task] for task in tasks}
            for rank, tasks in descriptor.producers_by_rank
        },
        "rendezvous_ready_ms": engine.completed_at[registration.join_task_id],
        "complete_ms": engine.completed_at[registration.completion_task_id],
        "byte_accounting": {
            "logical_payload_bytes_by_rank": dict(descriptor.to_dict()["payload_bytes_by_rank"]),
            "declared_physical_link_bytes": declared, "serviced_physical_link_bytes": serviced,
            "conserved": conserved,
            "policy": "explicit physical work charged once; payload, joins and gates add no wire bytes",
        },
    }


def collective_plan_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run a small explicit graph: all producers, collectives, then consumers."""
    data = _mapping(payload, {"schema_version", "name", "source", "resources", "producers", "collectives", "consumers"},
                    "collective_report_input_invalid")
    if (isinstance(data.get("schema_version"), bool)
            or not isinstance(data.get("schema_version"), int) or data.get("schema_version") != 1):
        raise A3InputError("collective_schema_version_unsupported")
    resources: dict[str, ResourceSpec] = {}
    for raw in _sequence(data.get("resources"), "collective_resources_invalid"):
        item = _mapping(raw, {"name", "capacity", "unit", "scope", "policy"}, "collective_resource_invalid")
        name = _text(item.get("name"), "collective_resource_name_invalid")
        if name in resources:
            raise A3InputError(f"collective_resource_duplicate:{name}")
        resources[name] = ResourceSpec(
            name, _number(item.get("capacity"), "collective_resource_capacity_invalid"),
            _text(item.get("unit"), "collective_resource_unit_invalid"),
            _text(item.get("scope", "shared"), "collective_resource_scope_invalid"),
            item.get("policy", "equal_share"),
        )
    engine = ResourceEngine(resources)
    for raw in _sequence(data.get("producers", ()), "collective_producers_invalid"):
        _register_work(engine, raw)
    registrations: dict[CollectiveKey, CollectiveRegistration] = {}
    for raw in _sequence(data.get("collectives", ()), "collective_descriptors_invalid"):
        descriptor = CollectiveDescriptor.from_mapping(raw)
        if descriptor.key in registrations:
            raise A3InputError("collective_identity_duplicate")
        registrations[descriptor.key] = register_collective(engine, descriptor)
    for raw in _sequence(data.get("consumers", ()), "collective_consumers_invalid"):
        item = _mapping(raw, {"task_id", "demands", "dependencies", "release_ms", "rank", "wait_for"},
                        "collective_consumer_invalid")
        rank = _integer(item.get("rank"), "collective_consumer_rank_invalid")
        deps: list[str] = []
        for ref in _sequence(item.get("wait_for", ()), "collective_consumer_wait_for_invalid"):
            key = CollectiveKey.from_mapping(ref)
            if key not in registrations:
                raise A3InputError("collective_consumer_identity_unknown:" + _canonical(key.to_dict()))
            deps.append(registrations[key].consumer_gate(rank))
        _register_work(engine, {key: value for key, value in item.items() if key not in {"rank", "wait_for"}}, deps=deps)
    execution = engine.run()
    if execution["remaining_tasks"] or not execution["drain_complete"]:
        raise A3InputError("collective_execution_incomplete")
    reports = [_report_collective(item, engine) for item in registrations.values()]
    declared: dict[str, float] = {}
    for report in reports:
        for name, amount in report["byte_accounting"]["declared_physical_link_bytes"].items():
            declared[name] = declared.get(name, 0.0) + amount
    return {
        "status": "modeled", "command": "collective-plan", "schema_version": 1,
        "name": _text(data.get("name", "collective-dag"), "collective_report_name_invalid"),
        "source": _text(data.get("source", "declared"), "collective_report_source_invalid"),
        "input_sha256": hashlib.sha256(_canonical(payload).encode()).hexdigest(),
        "effective_input": dict(payload),
        "scope": "explicit non-streaming participant rendezvous and physical link service; analytic, not NPU measurement",
        "limitations": ["no inferred transport algorithm or payload-to-wire conversion", "no endpoint local work inferred",
                        "no automatic predict, EP or DCP graph adaptation; no streaming chunk overlap"],
        "timing_scope": "declared_resource_dag_ms", "elapsed_ms": execution["makespan_ms"],
        "collectives": reports,
        "byte_accounting": {
            "declared_collective_physical_link_bytes": declared,
            "all_work_serviced_link_bytes": {name: amount for name, amount in execution["serviced"].items()
                                             if name.startswith("link:")},
            "collectives_conserved": all(report["byte_accounting"]["conserved"] for report in reports),
        },
        "execution": execution,
    }


__all__ = ["CollectiveKey", "CollectiveLinkStage", "CollectiveDescriptor", "CollectiveRegistration",
           "register_collective", "collective_plan_report"]
