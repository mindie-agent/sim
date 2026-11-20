"""Piecewise HCCL protocol-band selection and collective phase lowering.

[Role]
  Implement ``select_band`` over declared ``CollectiveBandSpec`` intervals
  (``[0, 65537)``, ``[65537, 2097152)``, ``[2097152, None)``), exact protocol
  framing, zero-payload rendezvous preservation, Ring AllReduce ``2(P-1)B/P``
  and AllToAllV integer peer-matrix verification, and shared-port Engine
  submission for multi-coset contention.

[Invariants]
  * Payload intervals under a fixed selector are disjoint; missing coverage or
    ambiguous matches raise explicit ``ModelingError``.
  * Zero-payload calls preserve caller rendezvous without emitting wire work.
  * Ring AllReduce lowers into dependent ``reduce_scatter`` then ``all_gather``
    phases with verified per-rank payload ``2 * (P - 1) * B / P``.
  * AllToAllV validates non-negative integer peer matrices, separates diagonal
    self-copy bytes from wire flows, and verifies row/column sums.
  * Startup latency is charged strictly once per ``band.startup_scope``
    (``collective``, ``phase``, or ``hop``); ``busbw`` is never reused as
    directed link bandwidth.

[Timeouts & Locks]
  Bounded group-size loop and in-memory band lookup; no I/O or locks.

[Side-Effect Policy]
  Pure planning helpers plus Engine task submission on caller-owned Engines.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware_links import CollectiveBandSpec
from ..communication import MessageFlow
from .ports import PairKey, PortKey, link_pair_key, link_port_key

MS_PER_S = 1000.0
SMALL_BAND_UPPER_EXCLUSIVE = 65537
MID_BAND_UPPER_EXCLUSIVE = 2097152
SUPPORTED_COLLECTIVES = frozenset({
    "allreduce", "reduce_scatter", "allgather", "alltoall", "alltoallv",
})


@dataclass(frozen=True, slots=True)
class LoweredTransfer:
    """One framed directed transfer within a lowered collective phase."""

    phase_name: str
    round_index: int
    source: int
    destination: int
    payload_bytes: float
    frames: int
    wire_bytes: float
    pair_keys: tuple[PairKey, ...]
    port_keys: tuple[PortKey, ...]
    resources: tuple[str, ...]
    propagation_ms: float


@dataclass(frozen=True, slots=True)
class LoweredPhase:
    """One causal phase (e.g. ``reduce_scatter`` or ``all_gather``)."""

    name: str
    rounds: int
    transfers: tuple[LoweredTransfer, ...]
    startup_us: float


@dataclass(frozen=True, slots=True)
class LoweredCollective:
    """Complete physical collective lowering with framing and port keys."""

    collective: str
    group: tuple[int, ...]
    band: CollectiveBandSpec
    communicator_id: str
    coset_id: str
    api_bytes_per_rank: float
    rendezvous_required: bool
    phases: tuple[LoweredPhase, ...]
    total_payload_bytes: float
    total_wire_bytes: float
    per_rank_payload_bytes: tuple[tuple[int, float], ...]
    self_copy_bytes_by_rank: tuple[tuple[int, int], ...]
    collective_startup_us: float

    @property
    def port_keys(self) -> frozenset[PortKey]:
        return frozenset(
            key
            for phase in self.phases
            for transfer in phase.transfers
            for key in transfer.port_keys
        )


def _normalize_collective_name(call) -> str:
    if isinstance(call, str):
        name = call.strip().lower()
    elif isinstance(call, Mapping):
        name = str(call.get("collective") or call.get("family") or "").strip().lower()
    else:
        meta = dict(getattr(call, "metadata", ()) or ())
        name = str(meta.get("collective") or getattr(call, "family", "")).strip().lower()
    aliases = {"all_reduce": "allreduce", "all_gather": "allgather",
               "all_to_all": "alltoall", "all_to_all_v": "alltoallv"}
    name = aliases.get(name, name)
    if name not in SUPPORTED_COLLECTIVES:
        raise ModelingError(f"collective_protocol_call_unsupported:{name}")
    return name


def _normalize_group(actual_group: Sequence[int]) -> tuple[int, ...]:
    group = tuple(int(rank) for rank in actual_group)
    if not group or len(set(group)) != len(group):
        raise ModelingError("collective_protocol_group_invalid")
    return group


def _infer_group_scope(group: tuple[int, ...], topology) -> str:
    if len(group) <= 1:
        return "package"
    if hasattr(topology, "scope_for_group"):
        return topology.scope_for_group(group)
    nodes = {topology.location(rank)["node"] for rank in group}
    if len(nodes) > 1:
        return "cross_node"
    pkgs = {(topology.location(rank)["node"], topology.location(rank)["package"])
            for rank in group}
    return "package" if len(pkgs) == 1 else "node"


def _link_bands(topology, link) -> tuple[CollectiveBandSpec, ...]:
    if hasattr(topology, "link_protocol_bands"):
        return tuple(topology.link_protocol_bands(link.name))
    return tuple(getattr(link, "protocol_bands", ()))


def _candidate_bands(group: tuple[int, ...], topology, scope: str) -> tuple[CollectiveBandSpec, ...]:
    collected: list[CollectiveBandSpec] = []
    seen_ids: set[int] = set()
    if len(group) >= 2:
        for index in range(len(group)):
            src = group[index]
            dst = group[(index + 1) % len(group)]
            if src == dst:
                continue
            try:
                path = topology.route(src, dst)
            except ValueError:
                continue
            for link_name, _, _ in path:
                link = topology.links.get(link_name)
                if link is None:
                    continue
                for band in _link_bands(topology, link):
                    if id(band) not in seen_ids:
                        seen_ids.add(id(band))
                        collected.append(band)
    if not collected:
        for link in topology.links.values():
            if topology.link_scope(link.name) != scope and scope != "any":
                continue
            for band in _link_bands(topology, link):
                if id(band) not in seen_ids:
                    seen_ids.add(id(band))
                    collected.append(band)
    return tuple(collected)


def select_band(
    call,
    actual_group: Sequence[int],
    topology,
    api_bytes_per_rank: int | float,
    stack: str = "default",
) -> CollectiveBandSpec:
    """Select the unique matching ``CollectiveBandSpec`` for a collective call."""
    collective = _normalize_collective_name(call)
    group = _normalize_group(actual_group)
    if (isinstance(api_bytes_per_rank, bool)
            or not isinstance(api_bytes_per_rank, (int, float))
            or not math.isfinite(float(api_bytes_per_rank))
            or float(api_bytes_per_rank) < 0):
        raise ModelingError(f"collective_api_bytes_invalid:{api_bytes_per_rank}")
    payload = float(api_bytes_per_rank)
    stack_key = str(stack or "default").strip() or "default"
    scope = _infer_group_scope(group, topology)
    candidates = _candidate_bands(group, topology, scope)

    matches: list[tuple[tuple[int, int, int], CollectiveBandSpec]] = []
    for band in candidates:
        if band.collective not in (collective, "all"):
            continue
        if band.stack_id not in (stack_key, "default"):
            continue
        if band.topology_scope not in (scope, "any"):
            continue
        if not (band.group_min <= len(group) <= band.group_max):
            continue
        if not band.matches_payload(payload):
            continue
        specificity = (
            1 if band.collective == collective else 0,
            1 if band.stack_id == stack_key else 0,
            1 if band.topology_scope == scope else 0,
        )
        matches.append((specificity, band))

    if not matches:
        raise ModelingError(
            f"collective_band_missing:{collective}:scope={scope}:"
            f"group={len(group)}:bytes={int(payload)}:stack={stack_key}"
        )
    best_rank = max(score for score, _ in matches)
    winners = [band for score, band in matches if score == best_rank]
    unique_winners = list(dict.fromkeys(winners))
    if len(unique_winners) > 1:
        raise ModelingError(
            f"collective_band_ambiguous:{collective}:scope={scope}:"
            f"bytes={int(payload)}:count={len(unique_winners)}"
        )
    return unique_winners[0]


def frame_wire_bytes(
    payload_bytes: int | float, band: CollectiveBandSpec
) -> tuple[int, float]:
    """Compute ``(frame_count, wire_bytes)`` under ``band`` framing."""
    payload = float(payload_bytes)
    if payload <= 0.0:
        return 0, 0.0
    frames = max(1, math.ceil(payload / float(band.payload_bytes_per_frame) - 1e-12))
    return frames, float(frames * band.wire_bytes_per_frame)


def _validate_peer_matrix(
    peer_matrix: Sequence[Sequence[int]], size: int
) -> tuple[tuple[int, ...], ...]:
    if len(peer_matrix) != size:
        raise ModelingError(
            f"alltoallv_peer_matrix_shape_invalid:{len(peer_matrix)}!={size}"
        )
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(peer_matrix):
        if len(row) != size:
            raise ModelingError(
                f"alltoallv_peer_matrix_row_invalid:{row_index}:{len(row)}!={size}"
            )
        parsed_row: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"alltoallv_peer_matrix_entry_invalid:{value}")
            parsed_row.append(value)
        rows.append(tuple(parsed_row))
    return tuple(rows)


def _build_transfer(
    phase_name: str,
    round_index: int,
    source: int,
    destination: int,
    payload_bytes: float,
    band: CollectiveBandSpec,
    topology,
) -> LoweredTransfer:
    frames, wire_bytes = frame_wire_bytes(payload_bytes, band)
    path = topology.route(source, destination)
    pair_keys: list[PairKey] = []
    port_keys: list[PortKey] = []
    resources: list[str] = []
    propagation_ms = 0.0
    for link_name, hop_src, hop_dst in path:
        link = topology.links[link_name]
        pair_keys.append(link_pair_key(link, hop_src, hop_dst))
        src_node = topology.location(hop_src)["node"]
        dst_node = topology.location(hop_dst)["node"]
        port_keys.append(link_port_key(link, hop_src, src_node, "tx"))
        port_keys.append(link_port_key(link, hop_dst, dst_node, "rx"))
        for res_name in topology.capacity_resources(link, hop_src, hop_dst):
            if res_name not in resources:
                resources.append(res_name)
        propagation_ms += link.propagation_us / MS_PER_S
    return LoweredTransfer(
        phase_name=phase_name,
        round_index=round_index,
        source=source,
        destination=destination,
        payload_bytes=float(payload_bytes),
        frames=frames,
        wire_bytes=wire_bytes,
        pair_keys=tuple(pair_keys),
        port_keys=tuple(port_keys),
        resources=tuple(resources),
        propagation_ms=propagation_ms,
    )


def lower_collective(
    call,
    actual_group: Sequence[int],
    topology,
    api_bytes_per_rank: int | float,
    *,
    stack: str = "default",
    peer_matrix: Sequence[Sequence[int]] | None = None,
    communicator_id: str = "",
    coset_id: str = "",
) -> LoweredCollective:
    """Lower a collective call into framed physical phases and port keys."""
    collective = _normalize_collective_name(call)
    group = _normalize_group(actual_group)
    band = select_band(collective, group, topology, api_bytes_per_rank, stack=stack)
    count = len(group)
    api_bytes = float(api_bytes_per_rank)
    collective_startup = band.startup_us if band.startup_scope == "collective" else 0.0
    phase_startup = band.startup_us if band.startup_scope == "phase" else 0.0

    self_copies: dict[int, int] = {rank: 0 for rank in group}
    per_rank_payload: dict[int, float] = {rank: 0.0 for rank in group}

    if collective == "alltoallv":
        if peer_matrix is None:
            raise ModelingError("alltoallv_requires_explicit_integer_peer_matrix")
        matrix = _validate_peer_matrix(peer_matrix, count)
        row_sums = [sum(row) for row in matrix]
        col_sums = [sum(matrix[r][c] for r in range(count)) for c in range(count)]
        if sum(row_sums) != sum(col_sums):
            raise ModelingError("alltoallv_peer_matrix_conservation_violation")
        transfers: list[LoweredTransfer] = []
        for s_idx, src in enumerate(group):
            self_copies[src] = matrix[s_idx][s_idx]
            for d_idx, dst in enumerate(group):
                if s_idx == d_idx:
                    continue
                amount = matrix[s_idx][d_idx]
                if amount == 0:
                    continue
                per_rank_payload[src] += float(amount)
                transfers.append(_build_transfer(
                    "alltoallv", 0, src, dst, float(amount), band, topology
                ))
        phases = (
            (LoweredPhase("alltoallv", 1, tuple(transfers), phase_startup),)
            if transfers else ()
        )
    elif api_bytes == 0.0 or count <= 1:
        phases = ()
    elif collective in {"allreduce", "reduce_scatter", "allgather"}:
        phase_names = (
            ("reduce_scatter", "all_gather")
            if collective == "allreduce"
            else (collective,)
        )
        rounds = count - 1
        chunk_payload = api_bytes / float(count)
        built_phases: list[LoweredPhase] = []
        for phase_name in phase_names:
            transfers = []
            for round_index in range(rounds):
                for idx, src in enumerate(group):
                    dst = group[(idx + 1) % count]
                    per_rank_payload[src] += chunk_payload
                    transfers.append(_build_transfer(
                        phase_name, round_index, src, dst,
                        chunk_payload, band, topology,
                    ))
            built_phases.append(
                LoweredPhase(phase_name, rounds, tuple(transfers), phase_startup)
            )
        if collective == "allreduce":
            expected_per_rank = 2.0 * (count - 1) * api_bytes / float(count)
            for rank in group:
                if not math.isclose(per_rank_payload[rank], expected_per_rank,
                                    rel_tol=1e-12, abs_tol=1e-9):
                    raise ModelingError(
                        f"ring_allreduce_payload_conservation_failed:{rank}"
                    )
        phases = tuple(built_phases)
    else:
        # Balanced alltoall
        per_peer = api_bytes / float(count - 1)
        transfers = []
        for s_idx, src in enumerate(group):
            for step in range(count - 1):
                dst = group[(s_idx + step + 1) % count]
                per_rank_payload[src] += per_peer
                transfers.append(_build_transfer(
                    "alltoall", step, src, dst, per_peer, band, topology
                ))
        phases = (LoweredPhase("alltoall", count - 1, tuple(transfers), phase_startup),)

    total_payload = sum(per_rank_payload.values())
    total_wire = sum(t.wire_bytes for phase in phases for t in phase.transfers)
    return LoweredCollective(
        collective=collective,
        group=group,
        band=band,
        communicator_id=str(communicator_id),
        coset_id=str(coset_id),
        api_bytes_per_rank=api_bytes,
        rendezvous_required=True,
        phases=phases,
        total_payload_bytes=total_payload,
        total_wire_bytes=total_wire,
        per_rank_payload_bytes=tuple((r, per_rank_payload[r]) for r in group),
        self_copy_bytes_by_rank=tuple((r, self_copies[r]) for r in group),
        collective_startup_us=collective_startup,
    )


def submit_lowered_collective(
    engine,
    name: str,
    lowered: LoweredCollective,
    ready_by_rank: Mapping[int, object],
    *,
    owner: str = "",
):
    """Submit a ``LoweredCollective`` onto ``engine`` honoring causal phases."""
    required = set(lowered.group)
    if set(ready_by_rank) != required:
        raise ModelingError("lowered_collective_ready_coverage_mismatch")
    rendezvous = engine.submit(
        f"{name}:rendezvous",
        after=tuple(ready_by_rank[r] for r in lowered.group),
        latency=lowered.collective_startup_us / MS_PER_S,
        owner=owner,
    )
    if not lowered.phases:
        return {rank: rendezvous for rank in lowered.group}

    hop_startup_ms = (
        lowered.band.startup_us / MS_PER_S
        if lowered.band.startup_scope == "hop"
        else 0.0
    )
    cursor = rendezvous
    for p_idx, phase in enumerate(lowered.phases):
        phase_gate = (
            engine.submit(
                f"{name}:p{p_idx}:{phase.name}:start",
                after=(cursor,),
                latency=phase.startup_us / MS_PER_S,
                owner=owner,
            )
            if phase.startup_us > 0.0
            else cursor
        )
        by_round: dict[int, list[LoweredTransfer]] = {}
        for transfer in phase.transfers:
            by_round.setdefault(transfer.round_index, []).append(transfer)
        round_gate = phase_gate
        for round_index in sorted(by_round):
            round_done = []
            for t_idx, transfer in enumerate(by_round[round_index]):
                demand = {res: transfer.wire_bytes for res in transfer.resources}
                service = engine.submit(
                    f"{name}:p{p_idx}:r{round_index}:t{t_idx}:svc",
                    demand,
                    after=(round_gate,),
                    owner=owner,
                    coupled=True,
                )
                arrival = engine.submit(
                    f"{name}:p{p_idx}:r{round_index}:t{t_idx}:arr",
                    after=(service,),
                    latency=transfer.propagation_ms + hop_startup_ms,
                    owner=owner,
                )
                round_done.append(arrival)
            round_gate = engine.submit(
                f"{name}:p{p_idx}:r{round_index}:done",
                after=tuple(round_done),
                owner=owner,
            )
        cursor = round_gate

    complete = engine.submit(f"{name}:complete", after=(cursor,), owner=owner)
    return {rank: complete for rank in lowered.group}
