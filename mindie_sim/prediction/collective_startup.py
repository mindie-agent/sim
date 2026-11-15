"""AllToAll/AllToAllV startup-model resolution.

[Role]
  Separate the declared startup model (how many synchronization rounds a
  collective model charges) from the physical peer schedule (which peer sends
  and receives actually execute).  Explicit ``steps``, operator metadata,
  canonical runtime mode and hardware protocol are resolved in one place.

[Invariants]
  * ``peer_phases`` is the number of remote peers ``G-1`` for ``G > 1``; every
    peer message remains represented independently of the startup model.
  * ``legacy``/``ring`` charge ``G-1`` startup rounds, ``log2`` charges
    ``ceil(log2(G))``, ``concurrent`` charges one and explicit ``steps`` must
    be an integer in ``[1, G-1]``.
  * Booleans, floats and non-finite values are rejected before any zero-volume
    early return.  A single-rank group has no legal ``steps`` declaration and
    resolves to a zero-round no-op.
  * ``topology`` mode requires a declared hardware protocol for the family; no
    SKU name, latency value or ``a5`` string implies one.

[Timeouts & Locks]
  Pure integer arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only; the caller owns topology and scheduling state.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError


ZERO: int = 0
ONE: int = 1
ALLTOALL_FAMILIES = frozenset({"alltoall", "alltoallv"})
STARTUP_MODES = frozenset({"legacy", "ring", "log2", "concurrent", "topology"})
MODE_ALIASES = {
    "legacy": "legacy",
    "ring": "ring",
    "log2": "log2",
    "direct": "concurrent",
    "concurrent": "concurrent",
    "topology": "topology",
}
ORIGIN_EXPLICIT = "explicit_steps"
ORIGIN_OPERATOR = "operator_metadata"
ORIGIN_RUNTIME = "runtime"
ORIGIN_TOPOLOGY = "topology_protocol"
ORIGIN_LEGACY = "legacy_default"


@dataclass(frozen=True)
class CollectiveStartup:
    """Resolved startup model for one all-to-all collective instance."""

    peer_phases: int
    startup_rounds: int
    mode: str
    origin: str


def _ceil_log2(value: int) -> int:
    """``ceil(log2(value))`` for a positive integer using bit length."""
    return (int(value) - ONE).bit_length()


def _rounds_for_mode(mode: str, group_size: int) -> int:
    if group_size <= ONE:
        return ZERO
    if mode in {"legacy", "ring"}:
        return group_size - ONE
    if mode == "log2":
        return _ceil_log2(group_size)
    if mode == "concurrent":
        return ONE
    raise ModelingError(f"collective_startup_mode_unsupported:{mode}")


def normalize_startup_mode(value: Any, *, origin: str) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"collective_startup_mode_invalid:{value!r}")
    mode = MODE_ALIASES.get(value.strip().lower())
    if mode is None:
        raise ModelingError(f"collective_startup_mode_unsupported:{value}")
    return mode


def validate_steps(value: Any, group_size: int, *, extra_peers: int = ZERO) -> int:
    """Validate an explicit startup round count against the peer range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingError(f"communication_round_count_invalid:{value}")
    upper = int(group_size) - ONE + int(extra_peers)
    if int(group_size) <= ONE or value < ONE or value > upper:
        raise ModelingError(f"communication_round_count_invalid:{value}")
    return int(value)


def _metadata_steps(metadata: Mapping[str, Any] | None) -> Any:
    if not metadata:
        return None
    return metadata.get("steps")


def resolve_collective_startup(
    *, family: str, group_size: int, metadata: Mapping[str, Any],
    runtime_mode: str, topology_protocols: Mapping[str, str],
) -> CollectiveStartup:
    """Resolve the all-to-all startup descriptor for one group."""
    name = str(family).strip().lower()
    if name not in ALLTOALL_FAMILIES:
        raise ModelingError(f"collective_startup_family_unsupported:{family}")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < ONE:
        raise ModelingError(f"collective_group_size_invalid:{group_size}")
    peers = group_size - ONE
    steps = _metadata_steps(metadata)
    if steps is not None:
        return CollectiveStartup(
            peers, validate_steps(steps, group_size), "explicit", ORIGIN_EXPLICIT)
    operator_mode = (metadata or {}).get("collective_startup_mode")
    if operator_mode is not None:
        mode = normalize_startup_mode(operator_mode, origin=ORIGIN_OPERATOR)
        if group_size <= ONE:
            return CollectiveStartup(ZERO, ZERO, mode, ORIGIN_OPERATOR)
        return CollectiveStartup(
            peers, _rounds_for_mode(mode, group_size), mode, ORIGIN_OPERATOR)
    mode = normalize_startup_mode(runtime_mode, origin=ORIGIN_RUNTIME)
    if group_size <= ONE:
        return CollectiveStartup(ZERO, ZERO, mode, ORIGIN_RUNTIME)
    if mode == "topology":
        protocols = {str(key).strip().lower(): str(value).strip().lower()
                     for key, value in (topology_protocols or {}).items()}
        protocol = protocols.get(name)
        if protocol is None:
            raise ModelingError(
                f"collective_startup_protocol_undeclared:{name}")
        resolved = normalize_startup_mode(protocol, origin=ORIGIN_TOPOLOGY)
        if resolved == "topology":
            raise ModelingError(
                f"collective_startup_protocol_undeclared:{name}")
        return CollectiveStartup(
            peers, _rounds_for_mode(resolved, group_size), resolved, ORIGIN_TOPOLOGY)
    return CollectiveStartup(
        peers, _rounds_for_mode(mode, group_size), mode, ORIGIN_RUNTIME)
