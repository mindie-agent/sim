"""Offered-capacity service clocks and semantic resource-role mapping.

[Role]
  Convert a fluid Engine service interval into four Roofline service clocks
  using per-task delivered work, the offered capacity before that task's own
  coupled throttle, and explicit blocking resources.

[Invariants]
  * Same-path resources (HBM, MTE2, L1) constrain parallel throughput by their
    maximum required time, not their sum.
  * Zero offered capacity is a wait on a recorded blocker; it is never divided
    by an epsilon or fabricated as memory stall.
  * Distinct serial transfer phases remain separate causal nodes at the caller.
  * Network endpoint DMA follows its declared program role; link service and
    hop propagation are communication.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only immutable results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from ..efficiency.exposed_breakdown import ExposedClocks, roofline_clocks


ZERO = 0.0
ROLE_CUBE = "cube"
ROLE_VECTOR = "vector"
ROLE_MEMORY = "memory"
ROLE_COMMUNICATION = "communication"
ROLE_CONTROL = "control"
CUBE_RESOURCES = frozenset({"cube", "matrix", "pipe_m"})
CUBE_PREFIXES = ("cube_", "matrix_", "pipe_m_")
MEMORY_RESOURCES = frozenset({
    "hbm", "memory", "l2", "l1", "l0a", "l0b", "l0c", "ub",
    "mte1", "mte2", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", "dma",
})
MEMORY_PREFIXES = (
    "hbm_", "memory_", "l2_", "l1_", "l0", "ub_", "mte", "fixpipe_", "dma_",
)
VECTOR_RESOURCES = frozenset({
    "vector", "scalar", "reg_alu_lanes", "simt", "pipe_v", "pipe_s",
})
VECTOR_PREFIXES = ("vec_", "vector_", "scalar_", "simt_", "pipe_v_", "pipe_s_")
COMMUNICATION_RESOURCES = frozenset({
    "link", "switch", "port", "pair", "d2h", "h2d", "rd2h", "rh2d", "host_rc", "host_solo", "transport",
})
COMMUNICATION_PREFIXES = (
    "link_", "switch_", "port_", "pair_", "d2h_", "h2d_", "rd2h_", "rh2d_", "host_rc_", "host_solo_", "transport_",
)
CONTROL_NAMESPACES = frozenset({"stars", "host", "scheduler"})
_DIRECTION_SUFFIXES = frozenset({"r", "w", "read", "write"})
ROLE_BY_PREFIX = {
    "memory@": ROLE_MEMORY,
    "link:": ROLE_COMMUNICATION,
}


@dataclass(frozen=True)
class IntervalClocks:
    """Roofline service clocks plus explicit blocking resources."""

    clocks: ExposedClocks
    blockers: tuple[str, ...] = ()
    offered: Mapping[str, float] | None = None

    @property
    def blocked(self) -> bool:
        return bool(self.blockers)


def service_role(name: str) -> str:
    """Semantic role of a physical resource or service name."""
    value = str(name).strip().lower()
    base = value.split("@", 1)[0]
    parts = [part for part in base.split(":") if part]
    if not parts:
        return ROLE_CONTROL
    if any(part in COMMUNICATION_RESOURCES for part in parts):
        return ROLE_COMMUNICATION
    if any(part in CONTROL_NAMESPACES for part in parts):
        return ROLE_CONTROL
    if len(parts) > 1 and parts[-1] in _DIRECTION_SUFFIXES:
        parts = parts[:-1]
    stem = parts[-1]
    if stem in MEMORY_RESOURCES or stem.startswith(MEMORY_PREFIXES):
        return ROLE_MEMORY
    if stem in CUBE_RESOURCES or stem.startswith(CUBE_PREFIXES):
        return ROLE_CUBE
    if stem in VECTOR_RESOURCES or stem.startswith(VECTOR_PREFIXES):
        return ROLE_VECTOR
    if stem in COMMUNICATION_RESOURCES or stem.startswith(COMMUNICATION_PREFIXES):
        return ROLE_COMMUNICATION
    for prefix, role in ROLE_BY_PREFIX.items():
        if value.startswith(prefix):
            return role
    return ROLE_CONTROL


def residual_capacity(
    declared: Mapping[str, float], used: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Capacity left after other tasks served on the same physical units."""
    used = dict(used or {})
    result = {}
    for name, capacity in dict(declared).items():
        value = float(capacity) - float(used.get(name, ZERO))
        result[name] = max(ZERO, value)
    return result


def interval_clocks(
    *, delivered_work: Mapping[str, float],
    offered_capacities: Mapping[str, float],
    blockers: Any = (),
) -> IntervalClocks:
    """Reduce one service interval onto the four Roofline service classes."""
    blockers = tuple(sorted({str(item) for item in (blockers or ())}))
    cube = vector = memory = communication = ZERO
    offered = {name: float(value) for name, value in dict(offered_capacities).items()}
    for name, amount in dict(delivered_work or {}).items():
        work = float(amount)
        if work <= ZERO:
            continue
        role = service_role(str(name))
        capacity = offered.get(str(name))
        if capacity is None:
            capacity = offered.get(role)
        if capacity is None or float(capacity) <= ZERO:
            blockers = tuple(sorted(set(blockers) | {str(name)}))
            continue
        service = work / float(capacity)
        if role == ROLE_CUBE:
            cube = max(cube, service)
        elif role == ROLE_VECTOR:
            vector = max(vector, service)
        elif role == ROLE_MEMORY:
            memory = max(memory, service)
        elif role == ROLE_COMMUNICATION:
            communication = max(communication, service)
        else:
            vector = max(vector, service)
    return IntervalClocks(
        roofline_clocks(cube_ms=cube, vector_ms=vector, memory_ms=memory,
                        communication_ms=communication),
        blockers, offered)


def bind_interval(
    *, interval: IntervalClocks, control_ms: float = ZERO,
    opaque_control_ms: float = ZERO,
) -> tuple[ExposedClocks, float, float]:
    """Add explicit control nodes and return the disclosed control portions."""
    if control_ms < ZERO or opaque_control_ms < ZERO:
        raise ModelingError("exposed_interval_control_invalid")
    clocks = interval.clocks
    if control_ms > ZERO:
        clocks = ExposedClocks(
            clocks.all_ms + control_ms,
            clocks.without_communication_ms + control_ms,
            clocks.compute_ms + control_ms,
            clocks.cube_ms,
        )
    return clocks, float(control_ms), float(opaque_control_ms)
