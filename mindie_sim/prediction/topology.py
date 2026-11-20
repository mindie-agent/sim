"""Thin compatibility re-export for ``mindie_sim.prediction.collectives``.

[Role]
  Preserve external caller imports of ``mindie_sim.prediction.topology`` after
  consolidating routing, port cosets, protocol bands, and transport plans into
  ``mindie_sim.prediction.collectives``.

[Invariants]
  * Re-exports every symbol historically published by ``prediction.topology``.

[Timeouts & Locks]
  Pure re-export shim; no I/O or locks.

[Side-Effect Policy]
  Side-effect-free import facade.
"""
from __future__ import annotations

from .communication import MessageFlow
from .collectives.topology import (
    _PORT_SEMANTICS,
    _SCOPE_PRIORITY,
    ChunkSequence,
    DEFAULT_MAX_PHYSICAL_CHUNKS,
    DIRECT_MESH_FAMILIES,
    DIRECT_MESH_MAX_PEERS,
    DIRECT_MESH_PHASES,
    FlowPath,
    Topology,
    TopologyCommunicationWork,
    TransportOptions,
    TransportPlan,
    TransportRun,
    _declared_peer_traffic,
    _is_cross_node_link,
    _link_scope,
    _link_text,
    _peer_declared_volume,
    _peer_totals_match,
    _transport_ready,
    chunk_count,
    chunk_size_at,
    chunk_sizes,
)

__all__ = [
    "ChunkSequence",
    "DEFAULT_MAX_PHYSICAL_CHUNKS",
    "DIRECT_MESH_FAMILIES",
    "DIRECT_MESH_MAX_PEERS",
    "DIRECT_MESH_PHASES",
    "FlowPath",
    "MessageFlow",
    "Topology",
    "TopologyCommunicationWork",
    "TransportOptions",
    "TransportPlan",
    "TransportRun",
    "_PORT_SEMANTICS",
    "_SCOPE_PRIORITY",
    "_declared_peer_traffic",
    "_is_cross_node_link",
    "_link_scope",
    "_link_text",
    "_peer_declared_volume",
    "_peer_totals_match",
    "_transport_ready",
    "chunk_count",
    "chunk_size_at",
    "chunk_sizes",
]
