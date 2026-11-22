"""Physical collective routing, port cosets, protocol bands, and transport plans.

[Role]
  Public package facade exporting ring/pairwise collective plans, physical
  ``Topology`` routing, chunk/credit ``TransportPlan``, canonical
  ``PairKey``/``PortKey``/``RootComplexKey`` contention maps, and piecewise
  ``select_band`` / ``lower_collective`` protocol lowering.

[Invariants]
  * Preserves every public symbol historically exported by
    ``mindie_sim.prediction.collectives`` and ``mindie_sim.prediction.topology``.

[Timeouts & Locks]
  Pure re-export module; no I/O or locks.

[Side-Effect Policy]
  Side-effect-free package facade.
"""
from __future__ import annotations

from .band_plan import (
    BandCollectiveGroup,
    BandCollectivePlan,
    api_bytes_from_link_volume,
    build_band_collective_plan,
)
from .plans import (
    DIRECT_MESH_FAMILIES,
    DIRECT_MESH_MAX_PEERS,
    DIRECT_MESH_PHASES,
    _declared_peer_traffic,
    _peer_declared_volume,
    _peer_totals_match,
)
from .ports import (
    PairKey,
    PortKey,
    RootComplexKey,
    SwitchKey,
    build_shared_port_resources,
    build_shared_switch_resources,
    find_shared_port_keys,
    find_shared_switch_keys,
    host_transfer_demand,
    link_pair_key,
    link_port_key,
    root_complex_resources_for_dies,
)
from .protocol import (
    MID_BAND_UPPER_EXCLUSIVE,
    SMALL_BAND_UPPER_EXCLUSIVE,
    LoweredCollective,
    LoweredHop,
    LoweredPhase,
    LoweredTransfer,
    frame_wire_bytes,
    lower_collective,
    select_band,
    submit_lowered_collective,
)
from .ring_pairwise import (
    OneShotCollective,
    OneShotGroup,
    OneShotPlan,
    PairwiseCollective,
    PairwiseGroup,
    PairwisePlan,
    RingCollective,
    RingGroup,
    RingPlan,
    _joint_complete,
    _ready_by_rank,
    plan_options,
)
from .topology import (
    _PORT_SEMANTICS,
    _SCOPE_PRIORITY,
    Topology,
    TopologyCommunicationWork,
    _is_cross_node_link,
    _link_scope,
    _link_text,
)
from .transport import (
    DEFAULT_MAX_PHYSICAL_CHUNKS,
    ChunkSequence,
    FlowPath,
    TransportOptions,
    TransportPlan,
    TransportRun,
    _transport_ready,
    chunk_count,
    chunk_size_at,
    chunk_sizes,
)

__all__ = [
    "BandCollectiveGroup",
    "BandCollectivePlan",
    "ChunkSequence",
    "DEFAULT_MAX_PHYSICAL_CHUNKS",
    "DIRECT_MESH_FAMILIES",
    "DIRECT_MESH_MAX_PEERS",
    "DIRECT_MESH_PHASES",
    "FlowPath",
    "LoweredCollective",
    "LoweredHop",
    "LoweredPhase",
    "LoweredTransfer",
    "MID_BAND_UPPER_EXCLUSIVE",
    "OneShotCollective",
    "OneShotGroup",
    "OneShotPlan",
    "PairKey",
    "PairwiseCollective",
    "PairwiseGroup",
    "PairwisePlan",
    "PortKey",
    "RingCollective",
    "RingGroup",
    "RingPlan",
    "RootComplexKey",
    "SMALL_BAND_UPPER_EXCLUSIVE",
    "SwitchKey",
    "Topology",
    "TopologyCommunicationWork",
    "TransportOptions",
    "TransportPlan",
    "TransportRun",
    "_PORT_SEMANTICS",
    "_SCOPE_PRIORITY",
    "_declared_peer_traffic",
    "_is_cross_node_link",
    "_joint_complete",
    "_link_scope",
    "_link_text",
    "_peer_declared_volume",
    "_peer_totals_match",
    "_ready_by_rank",
    "_transport_ready",
    "api_bytes_from_link_volume",
    "build_band_collective_plan",
    "build_shared_port_resources",
    "build_shared_switch_resources",
    "chunk_count",
    "chunk_size_at",
    "chunk_sizes",
    "find_shared_port_keys",
    "find_shared_switch_keys",
    "frame_wire_bytes",
    "host_transfer_demand",
    "link_pair_key",
    "link_port_key",
    "lower_collective",
    "plan_options",
    "root_complex_resources_for_dies",
    "select_band",
    "submit_lowered_collective",
]
