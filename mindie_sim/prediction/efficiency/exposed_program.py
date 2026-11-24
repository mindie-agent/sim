"""Executable program role and overlap descriptors for attribution.

[Role]
  Attach semantic service roles and independent/concurrent overlap structure to
  the existing Stage/program declarations so the exposed observer can price a
  realized schedule without re-simulating it.

[Invariants]
  * Cube, vector, memory and communication work map to semantic resource roles,
    never op-name substrings.
  * A coupled Roofline stage uses its four demands concurrently; distinct serial
    stages are joined by the observer's causal edges.
  * Host launch/minimum latency is explicit control, disclosed separately from
    measured AIV arithmetic.

[Timeouts & Locks]
  Pure record construction; no I/O or locks.

[Side-Effect Policy]
  Read-only immutable descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.runtime.exposed_intervals import service_role


ROLE_CUBE = "cube"
ROLE_VECTOR = "vector"
ROLE_MEMORY = "memory"
ROLE_COMMUNICATION = "communication"
ROLE_CONTROL = "control"
ROLE_COUPLED = "coupled"
ROLES = frozenset({
    ROLE_CUBE, ROLE_VECTOR, ROLE_MEMORY, ROLE_COMMUNICATION, ROLE_CONTROL,
    ROLE_COUPLED,
})
MEMORY_RESOURCES = frozenset({"hbm", "mte2", "mte2_aic", "mte2_aiv", "mte3"})
COMMUNICATION_RESOURCES = frozenset({"link"})
VECTOR_RESOURCES = frozenset({"vector", "scalar"})


@dataclass(frozen=True)
class StageDescriptor:
    """Declared service-class durations for one executable stage."""

    role: str
    cube_ms: float = 0.0
    vector_ms: float = 0.0
    memory_ms: float = 0.0
    communication_ms: float = 0.0
    control_ms: float = 0.0
    opaque_control_ms: float = 0.0
    serial_after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ModelingError(f"exposed_stage_role_unsupported:{self.role}")
        for name in ("cube_ms", "vector_ms", "memory_ms", "communication_ms",
                     "control_ms", "opaque_control_ms"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ModelingError(f"exposed_stage_duration_negative:{name}")


MATRIX_PREFIXES = ("matrix@", "cube@")
VECTOR_PREFIXES = ("vector@", "scalar@")
MEMORY_PREFIXES = ("mte", "fixpipe", "l2@", "ub", "l0")


def _resource_role(name: str) -> str:
    """Semantic role of one scoped physical resource or legacy alias."""
    role = service_role(name)
    if role != ROLE_CONTROL:
        return role
    if name.startswith(MATRIX_PREFIXES):
        return ROLE_CUBE
    if name.startswith(VECTOR_PREFIXES):
        return ROLE_VECTOR
    if name.startswith(MEMORY_PREFIXES):
        return ROLE_MEMORY
    return ROLE_CONTROL


def _role_times(work: Mapping[str, float], rates: Mapping[str, float]) -> dict[str, float]:
    cube = vector = memory = communication = 0.0
    for name, amount in dict(work or {}).items():
        value = float(amount)
        if value <= 0.0:
            continue
        rate = float(dict(rates or {}).get(name, 0.0))
        if rate <= 0.0:
            raise ModelingError(f"exposed_stage_rate_missing:{name}")
        service = value / rate
        role = _resource_role(str(name))
        if role == ROLE_CUBE:
            cube = max(cube, service)
        elif role == ROLE_VECTOR:
            vector = max(vector, service)
        elif role == ROLE_MEMORY:
            memory = max(memory, service)
        elif role == ROLE_COMMUNICATION:
            communication = max(communication, service)
        else:
            # Host launch/control is a vector/control service node.
            vector = max(vector, service)
    return {"cube_ms": cube, "vector_ms": vector,
            "memory_ms": memory, "communication_ms": communication}


def descriptor_for_stage(
    stage: Any, rates: Mapping[str, float], *,
    minimum_ms: float = 0.0, minimum_opaque: bool = False,
) -> StageDescriptor:
    """Derive a stage descriptor from its declared work and service rates."""
    times = _role_times(getattr(stage, "work", {}) or {}, rates)
    latency = float(getattr(stage, "latency", 0.0) or 0.0)
    control = float(minimum_ms or 0.0) + max(0.0, latency)
    coupled = bool(getattr(stage, "coupled", False))
    if coupled:
        role = ROLE_COUPLED
    elif times["cube_ms"] > 0.0:
        role = ROLE_CUBE
    elif times["vector_ms"] > 0.0:
        role = ROLE_VECTOR
    elif times["memory_ms"] > 0.0:
        role = ROLE_MEMORY
    elif times["communication_ms"] > 0.0:
        role = ROLE_COMMUNICATION
    else:
        role = ROLE_CONTROL
    return StageDescriptor(
        role=role, control_ms=control,
        opaque_control_ms=(control if minimum_opaque else 0.0),
        serial_after=tuple(getattr(stage, "after", ()) or ()),
        **times)


def program_descriptors(
    program: Any, rates: Mapping[str, float], *,
    minimum_ms: float = 0.0,
) -> tuple[StageDescriptor, ...]:
    """Describe every stage of one executable program in declaration order."""
    stages = tuple(getattr(program, "stages", ()) or ())
    total_minimum = float(minimum_ms or 0.0)
    descriptors = []
    for index, stage in enumerate(stages):
        descriptors.append(descriptor_for_stage(
            stage, rates,
            minimum_ms=(total_minimum if index == 0.0 else 0.0),
            minimum_opaque=bool(getattr(program, "minimum_provenance", None)),
        ))
    if not stages and total_minimum > 0.0:
        descriptors.append(StageDescriptor(
            ROLE_CONTROL, control_ms=total_minimum, opaque_control_ms=total_minimum))
    return tuple(descriptors)


def declarations_from_program(
    program: Any, rates: Mapping[str, float], *, minimum_ms: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    """Serializable form of every stage descriptor."""
    return tuple({
        "role": item.role,
        **({"cube_ms": item.cube_ms} if item.cube_ms else {}),
        **({"vector_ms": item.vector_ms} if item.vector_ms else {}),
        **({"memory_ms": item.memory_ms} if item.memory_ms else {}),
        **({"communication_ms": item.communication_ms} if item.communication_ms else {}),
        **({"control_ms": item.control_ms} if item.control_ms else {}),
        **({"opaque_control_ms": item.opaque_control_ms}
           if item.opaque_control_ms else {}),
        **({"serial_after": list(item.serial_after)} if item.serial_after else {}),
    } for item in program_descriptors(program, rates, minimum_ms=minimum_ms))
