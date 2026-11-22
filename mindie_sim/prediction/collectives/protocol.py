"""Piecewise HCCL protocol-band selection and collective phase lowering.

[Role] Select protocol bands and lower collective executors onto physical links.
[Invariants] Band selection is unique. Preserve API byte conventions, self
  copies and algorithm-specific wire work. Each hop follows its predecessor
  and shares the declared receive credit pool.
[Timeouts & Locks] Bounded group/chunk loops and in-memory band lookup.
[Side-Effect Policy] Pure planning plus task/pool submission on caller Engines.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware_links import CollectiveBandSpec
from ..engine import Allocation, Resource
from mindie_sim.prediction.operators.collectives.hccl_nhr import rounds as nhr_rounds
from .ports import PairKey, PortKey, link_pair_key, link_port_key
from .transport import DEFAULT_MAX_PHYSICAL_CHUNKS, chunk_sizes

MS_PER_S = 1000.0
SMALL_BAND_UPPER_EXCLUSIVE = 65537
MID_BAND_UPPER_EXCLUSIVE = 2097152
TWO_RANKS = 2
KIB = 1024
MIB = KIB ** 2
K8_RANKS = 8
A3_NODE_RANKS = 2 * K8_RANKS
A2_SMALL_COUNT_BYTES = 16 * MIB
A3_AIV_NODE_BYTES = 512 * KIB
A3_AIV_POD_BYTES = 8 * MIB
A3_RS_MESH_BYTES = 512 * KIB
A3_RS_LARGE_MESH_BYTES = 1 * MIB
A5_CCU_ONE_SHOT_SHARD_BYTES = 16 * KIB
A5_AICPU_ONE_SHOT_BYTES = 8 * MIB
A5_AICPU_ALLREDUCE_CHUNK_BYTES = 16 * MIB
A5_AICPU_RS_CHUNK_BYTES = 32 * MIB
HIERARCHICAL_2D_SLICE_BYTES = 16 * MIB
HCCL_DEFAULT_BUFFSIZE_BYTES = 200 * MIB
SCRATCH_STAGING_HBM_PASSES = 4  # Pack (R+W) + unpack (R+W) through HCCL scratch buffer.
LOCAL_REDUCE_HBM_PASSES = 3  # Two operand reads + one reduced result write per binary add.
P18_ALLREDUCE_EMBEDDED_HBM_PER_WIRE = 1.5  # 2-rank AllReduce embeds 3*(N/2) HBM per 1*N wire.
UB_DIE_INJECTION_CEILING_BYTES_PER_MS = 280_000_000.0
UB_INJECTION_CEILING_MIN_RANKS = 5
DOUBLE_BUFFER_CREDIT_FRAMES = 2
_HCOMM_EXPANSIONS = {
    "aiv": "aiv", "ccu_ms": "ccu_ms", "ai_cpu": "aicpu", "aicpu": "aicpu", "aicpu_ts": "aicpu", "nhr": "nhr",
}
_PHYSICAL_BAND_LIMITS = {"hccs_k8": K8_RANKS, "ub_mesh": K8_RANKS, "hccs_sw": A3_NODE_RANKS, "sio": TWO_RANKS}
_MESH_ALGORITHMS = frozenset({
    "direct", "direct_mesh", "pairwise", "mesh", "small_count_mesh", "one_shot_mesh",
    "two_shot_mesh", "chunked_two_shot_mesh", "chunked_mesh",
})
SUPPORTED_COLLECTIVES = frozenset({
    "allreduce", "reduce_scatter", "allgather", "alltoall", "alltoallv",
})
SUPPORTED_BAND_ALGORITHMS = frozenset({
    "ring", "double_ring", "nhr", "halving_doubling", "recursive_halving_doubling", "tree", "hierarchical_2d",
}) | _MESH_ALGORITHMS


@dataclass(frozen=True, slots=True)
class LoweredHop:
    """One physical hop within a lowered directed transfer."""

    link_name: str
    source: int
    destination: int
    pair_key: PairKey
    tx_port: PortKey
    rx_port: PortKey
    resources: tuple[str, ...]
    propagation_ms: float
    switch_propagation_ms: float = 0.0


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
    hops: tuple[LoweredHop, ...]
    slice_index: int = 0
    flow_cap_bytes_per_ms: float | None = None

    @property
    def pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(hop.pair_key for hop in self.hops)

    @property
    def port_keys(self) -> tuple[PortKey, ...]:
        keys: list[PortKey] = []
        for hop in self.hops:
            keys.append(hop.tx_port)
            keys.append(hop.rx_port)
        return tuple(keys)

    @property
    def resources(self) -> tuple[str, ...]:
        seen: list[str] = []
        for hop in self.hops:
            for res in hop.resources:
                if res not in seen:
                    seen.append(res)
        return tuple(seen)

    @property
    def propagation_ms(self) -> float:
        return sum(hop.propagation_ms for hop in self.hops)


@dataclass(frozen=True, slots=True)
class LoweredPhase:
    """One causal phase (e.g. ``reduce_scatter``, ``all_gather``, ``direct``)."""

    name: str
    rounds: int
    transfers: tuple[LoweredTransfer, ...]
    startup_us: float
    pipelined_slices: bool = False


@dataclass(frozen=True, slots=True)
class LoweredCollective:
    """Complete physical collective lowering with framing, hops, and port keys."""

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
    self_copy_bytes_by_rank: tuple[tuple[int, float], ...]
    collective_startup_us: float
    uses_calibrated_startup: bool = False
    die_injection_ceiling_bytes_per_ms: float | None = None
    local_hbm_bytes_per_rank: float = 0.0

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
    if len(group) >= TWO_RANKS:
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


def _package_partition(group: tuple[int, ...], topology) -> tuple[tuple[int, ...], ...]:
    """Partition ``group`` into equal-sized multi-die packages if uniform."""
    if not hasattr(topology, "_package_key") or len(group) < TWO_RANKS * TWO_RANKS:
        return ()
    by_pkg: dict[tuple[int, int], list[int]] = {}
    for rank in group:
        by_pkg.setdefault(topology._package_key(rank), []).append(rank)
    if len(by_pkg) < TWO_RANKS:
        return ()
    sizes = {len(members) for members in by_pkg.values()}
    if len(sizes) != 1 or next(iter(sizes)) < TWO_RANKS:
        return ()
    return tuple(tuple(members) for members in by_pkg.values())


def _hcomm_algorithm(fabric, collective, ranks, size, expansion, *, is_multi_package=False):
    """Select the modeled HCOMM Op-Base SUM paths from the vol. 8 source revision."""
    if fabric == "hccs_k8" and expansion in (None, "aiv"):
        if collective == "allreduce":
            return "small_count_mesh" if size < A2_SMALL_COUNT_BYTES else "mesh"
        if collective in {"allgather", "reduce_scatter"}:
            return "mesh"
        if collective == "alltoall":
            return "direct_mesh"
    if fabric in {"hccs_sw", "hccs_optical"}:
        if expansion == "nhr":
            return "nhr"
        if expansion is None and is_multi_package and collective in {"allreduce", "allgather", "reduce_scatter"}:
            logical_size = size * float(ranks) if collective == "allgather" else size
            if logical_size >= MID_BAND_UPPER_EXCLUSIVE:
                return "double_ring"
            return "one_shot_mesh" if collective == "allreduce" else "mesh"
        if expansion is None and collective == "alltoall":
            return "direct_mesh"
        if collective == "allreduce" and expansion in (None, "aiv", "aicpu"):
            aiv = size <= A3_AIV_NODE_BYTES if ranks <= K8_RANKS else size < A3_AIV_POD_BYTES
            if aiv and expansion != "aicpu":
                return "one_shot_mesh"
            return "mesh" if ranks & (ranks - 1) else "nhr"
        if fabric == "hccs_sw" and collective == "reduce_scatter" and expansion in (None, "aicpu"):
            if size <= A3_RS_MESH_BYTES:
                return "mesh"
            if size <= A3_RS_LARGE_MESH_BYTES:
                return "chunked_mesh"
            return "ring"
    if fabric == "ub_mesh":
        if collective == "allreduce":
            if expansion in (None, "ccu_ms"):
                return "one_shot_mesh" if size <= ranks * A5_CCU_ONE_SHOT_SHARD_BYTES else "mesh"
            if expansion == "aicpu":
                if size <= A5_AICPU_ONE_SHOT_BYTES:
                    return "one_shot_mesh"
                if size * K8_RANKS > A5_AICPU_ALLREDUCE_CHUNK_BYTES * ranks ** 2:
                    return "chunked_two_shot_mesh"
                return "two_shot_mesh"
        if collective in {"allgather", "reduce_scatter"} and expansion in (None, "ccu_ms"):
            return "mesh"
        if collective == "alltoall" and expansion in (None, "ccu_ms"):
            return "direct_mesh"
        if collective == "reduce_scatter" and expansion == "aicpu":
            return ("chunked_mesh" if size * K8_RANKS ** 2 > A5_AICPU_RS_CHUNK_BYTES * ranks ** 2
                    else "mesh")
    return None


def select_band(
    call,
    actual_group: Sequence[int],
    topology,
    api_bytes_per_rank: int | float,
    stack: str = "default",
    *,
    expansion_mode: str | None = None,
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
    exp_key = (None if expansion_mode is None
               else str(expansion_mode).strip().lower() or None)
    scope = _infer_group_scope(group, topology)
    candidates = _candidate_bands(group, topology, scope)
    is_multi_pkg = bool(
        getattr(topology, "_has_primitive_table", False)
        and _package_partition(group, topology)
    )

    matches: list[tuple[tuple[int, int, int, int], CollectiveBandSpec]] = []
    physical_limit = None
    for band in candidates:
        if band.collective not in (collective, "all"):
            continue
        if band.stack_id not in (stack_key, "default"):
            continue
        if band.topology_scope not in (scope, "any"):
            continue
        if not band.matches_payload(payload):
            continue
        fabric = band.protocol.rsplit("_", 1)[0]
        algorithm = None
        if band.algorithm == "ring" and band.stack_id == "default" and (exp_key is None or exp_key in _HCOMM_EXPANSIONS):
            algorithm = _hcomm_algorithm(
                fabric, collective, len(group), payload, _HCOMM_EXPANSIONS.get(exp_key),
                is_multi_package=is_multi_pkg,
            )
        explicit_executor = exp_key in _HCOMM_EXPANSIONS and algorithm is not None
        if exp_key is not None and band.expansion_mode != exp_key and not explicit_executor:
            continue
        limit = _PHYSICAL_BAND_LIMITS.get(fabric)
        if limit is not None and len(group) > limit:
            physical_limit = limit
            continue
        group_max = min(band.group_max, limit) if limit is not None else band.group_max
        if not band.group_min <= len(group) <= group_max:
            continue
        specificity = (
            1 if band.collective == collective else 0,
            1 if band.stack_id == stack_key else 0,
            1 if band.topology_scope == scope else 0,
            1 if (exp_key is not None and band.expansion_mode == exp_key) else 0,
        )
        if group_max != band.group_max or algorithm is not None:
            band = replace(band, group_max=group_max, algorithm=algorithm or band.algorithm,
                           expansion_mode=exp_key if explicit_executor else band.expansion_mode)
        matches.append((specificity, band))

    if not matches:
        if physical_limit is not None:
            raise ModelingError(f"collective_band_group_exceeds_physical_topology:{len(group)}>{physical_limit}")
        raise ModelingError(
            f"collective_band_missing:{collective}:scope={scope}:"
            f"group={len(group)}:bytes={int(payload)}:stack={stack_key}"
            + (f":expansion={exp_key}" if exp_key else "")
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
    """Compute ``(frame_count, wire_bytes)`` under ``band`` MTU + header framing."""
    payload = float(payload_bytes)
    if payload <= 0.0:
        return 0, 0.0
    frames = max(1, math.ceil(payload / float(band.payload_bytes_per_frame) - 1e-12))
    header_per_frame = max(
        0, int(band.wire_bytes_per_frame) - int(band.payload_bytes_per_frame)
    )
    return frames, float(payload + frames * header_per_frame)


def _validate_nonneg_int_seq(seq: Sequence[int], size: int, label: str) -> tuple[int, ...]:
    if len(seq) != size:
        raise ModelingError(f"alltoallv_{label}_length_invalid:{len(seq)}!={size}")
    parsed: list[int] = []
    for val in seq:
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ModelingError(f"alltoallv_{label}_entry_invalid:{val}")
        parsed.append(val)
    return tuple(parsed)


def _validate_alltoallv_inputs(
    peer_matrix: Sequence[Sequence[int]] | None,
    size: int,
    api_bytes: float,
    send_splits: Sequence[int] | None,
    recv_splits: Sequence[int] | None,
) -> tuple[tuple[int, ...], ...]:
    if peer_matrix is None:
        raise ModelingError("alltoallv_requires_explicit_integer_peer_matrix")
    if len(peer_matrix) != size:
        raise ModelingError(f"alltoallv_peer_matrix_shape_invalid:{len(peer_matrix)}!={size}")
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(peer_matrix):
        if len(row) != size:
            raise ModelingError(f"alltoallv_peer_matrix_row_invalid:{row_index}:{len(row)}!={size}")
        parsed_row: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"alltoallv_peer_matrix_entry_invalid:{value}")
            parsed_row.append(value)
        rows.append(tuple(parsed_row))
    matrix = tuple(rows)
    row_sums = tuple(sum(row) for row in matrix)
    col_sums = tuple(sum(matrix[r][c] for r in range(size)) for c in range(size))
    if send_splits is not None:
        declared_send = _validate_nonneg_int_seq(send_splits, size, "send_splits")
        if row_sums != declared_send:
            raise ModelingError(f"alltoallv_send_splits_mismatch:{row_sums}!={declared_send}")
    if recv_splits is not None:
        declared_recv = _validate_nonneg_int_seq(recv_splits, size, "recv_splits")
        if col_sums != declared_recv:
            raise ModelingError(f"alltoallv_recv_splits_mismatch:{col_sums}!={declared_recv}")
    total_matrix = sum(row_sums)
    if api_bytes == 0.0 and total_matrix > 0:
        raise ModelingError(f"alltoallv_api_bytes_zero_with_nonzero_matrix:{total_matrix}")
    if api_bytes > 0.0 and total_matrix == 0:
        raise ModelingError(f"alltoallv_api_bytes_nonzero_with_zero_matrix:{api_bytes}")
    if total_matrix > 0 and send_splits is not None:
        mean_send = total_matrix / float(size)
        valid_api = (
            math.isclose(api_bytes, float(max(row_sums)), rel_tol=1e-9, abs_tol=1e-6)
            or math.isclose(api_bytes, mean_send, rel_tol=1e-9, abs_tol=1e-6)
            or math.isclose(api_bytes, float(total_matrix), rel_tol=1e-9, abs_tol=1e-6)
        )
        if not valid_api:
            raise ModelingError(f"alltoallv_api_bytes_splits_mismatch:{api_bytes}")
    return matrix


def _build_transfer(
    phase_name: str, round_index: int, source: int, destination: int,
    payload_bytes: float, band: CollectiveBandSpec, topology,
    *, slice_index: int = 0,
) -> LoweredTransfer:
    frames, wire_bytes = frame_wire_bytes(payload_bytes, band)
    path = topology.route(source, destination)
    hops: list[LoweredHop] = []
    flow_caps: list[float] = []
    cap_fn = getattr(topology, "single_peer_flow_cap_bytes_per_ms", None)
    for link_name, hop_src, hop_dst in path:
        link = topology.links[link_name]
        if cap_fn is not None:
            cap_val = cap_fn(link_name)
            if cap_val is not None:
                flow_caps.append(float(cap_val))
        if hasattr(topology, "pair_key"):
            pkey = topology.pair_key(link, hop_src, hop_dst)
            tx_key = topology.port_key(link, hop_src, "tx")
            rx_key = topology.port_key(link, hop_dst, "rx")
        else:
            pkey = link_pair_key(link, hop_src, hop_dst)
            tx_key = link_port_key(link, hop_src, topology.location(hop_src)["node"], "tx")
            rx_key = link_port_key(link, hop_dst, topology.location(hop_dst)["node"], "rx")
        hop_res = tuple(dict.fromkeys(topology.capacity_resources(link, hop_src, hop_dst)))
        switch_prop_ms = (
            float(getattr(link, "switch_hops", 0) or 0)
            * float(getattr(link, "switch_hop_latency_us", 0.0) or 0.0)
        ) / MS_PER_S
        hops.append(LoweredHop(
            link_name=link_name,
            source=int(hop_src),
            destination=int(hop_dst),
            pair_key=pkey,
            tx_port=tx_key,
            rx_port=rx_key,
            resources=hop_res,
            propagation_ms=link.propagation_us / MS_PER_S,
            switch_propagation_ms=switch_prop_ms,
        ))
    return LoweredTransfer(
        phase_name=phase_name, round_index=round_index,
        source=int(source), destination=int(destination),
        payload_bytes=float(payload_bytes), frames=frames,
        wire_bytes=wire_bytes, hops=tuple(hops),
        slice_index=int(slice_index),
        flow_cap_bytes_per_ms=min(flow_caps) if flow_caps else None,
    )


def _subgroup_mesh_transfers(
    phase_name: str, subgroups: Sequence[tuple[int, ...]], per_peer_bytes: float,
    band: CollectiveBandSpec, topology, per_rank_payload: dict[int, float],
    *, slice_index: int,
) -> list[LoweredTransfer]:
    out: list[LoweredTransfer] = []
    for sub in subgroups:
        for src in sub:
            for dst in sub:
                if src == dst:
                    continue
                per_rank_payload[src] += per_peer_bytes
                out.append(_build_transfer(
                    phase_name, 0, src, dst, per_peer_bytes, band, topology,
                    slice_index=slice_index,
                ))
    return out


def _lower_hierarchical_2d(
    collective: str, group: tuple[int, ...], api_bytes: float,
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
) -> tuple[LoweredPhase, ...]:
    """Lower 2-level intra-package SIO + inter-package HCCS_SW pipelined hierarchy."""
    pkgs = _package_partition(group, topology)
    if not pkgs:
        return ()
    n_pkg, n_die = len(pkgs), len(pkgs[0])
    inter_groups = tuple(tuple(pkg[slot] for pkg in pkgs) for slot in range(n_die))
    logical_bytes = api_bytes * float(len(group)) if collective == "allgather" else api_bytes
    n_slices = min(
        DEFAULT_MAX_PHYSICAL_CHUNKS,
        max(1, math.ceil(logical_bytes / float(HIERARCHICAL_2D_SLICE_BYTES))),
    )
    slice_api = api_bytes / float(n_slices)
    if collective == "allreduce":
        stage_specs = (
            ("intra_rs", pkgs, slice_api / float(n_die)),
            ("inter_rs", inter_groups, slice_api / float(n_die * n_pkg)),
            ("inter_ag", inter_groups, slice_api / float(n_die * n_pkg)),
            ("intra_ag", pkgs, slice_api / float(n_die)),
        )
    elif collective == "allgather":
        stage_specs = (
            ("intra_ag", pkgs, slice_api),
            ("inter_ag", inter_groups, slice_api * float(n_die)),
        )
    else:
        stage_specs = (
            ("inter_rs", inter_groups, slice_api / float(n_pkg)),
            ("intra_rs", pkgs, slice_api / float(n_pkg * n_die)),
        )
    phases: list[LoweredPhase] = []
    for s_name, subgroups, peer_bytes in stage_specs:
        transfers: list[LoweredTransfer] = []
        for s_idx in range(n_slices):
            transfers.extend(_subgroup_mesh_transfers(
                s_name, subgroups, peer_bytes, band, topology, per_rank_payload,
                slice_index=s_idx,
            ))
        phases.append(LoweredPhase(
            s_name, 1, tuple(transfers), phase_startup,
            pipelined_slices=n_slices > 1,
        ))
    return tuple(phases)


def _lower_ring_or_alg(
    collective: str, group: tuple[int, ...], api_bytes: float,
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
) -> tuple[LoweredPhase, ...]:
    count = len(group)
    alg = str(band.algorithm).strip().lower()
    if alg not in SUPPORTED_BAND_ALGORITHMS:
        raise ModelingError(f"collective_band_algorithm_unsupported:{band.algorithm}")
    phase_names = ("reduce_scatter", "all_gather") if collective == "allreduce" else (collective,)
    shard = api_bytes if collective == "allgather" else api_bytes / count
    if alg == "hierarchical_2d":
        hier_phases = _lower_hierarchical_2d(
            collective, group, api_bytes, band, topology, phase_startup, per_rank_payload
        )
        if hier_phases:
            return hier_phases
    if alg in _MESH_ALGORITHMS:
        single_exchange = collective == "allreduce" and (
            alg in {"one_shot_mesh", "small_count_mesh"} or (alg == "direct" and count == TWO_RANKS)
        )
        if single_exchange:
            phase_names = ("direct" if alg == "direct" else "all_exchange",)
            shard = api_bytes
        phases = []
        for pname in phase_names:
            rounds = count - 1 if pname == "reduce_scatter" and alg.startswith("chunked_") else 1
            amount = shard / rounds
            transfers = tuple(_build_transfer(pname, step, src, dst, amount, band, topology)
                              for step in range(rounds) for src in group for dst in group if src != dst)
            for transfer in transfers:
                per_rank_payload[transfer.source] += transfer.payload_bytes
            phases.append(LoweredPhase(pname, rounds, transfers, phase_startup))
        return tuple(phases)
    if alg == "nhr":
        if count & (count - 1):
            raise ModelingError("collective_nhr_requires_power_of_two_ranks")
        phases = []
        for pname in phase_names:
            transfers = []
            for index, src in enumerate(group):
                schedule = nhr_rounds(count, index, pname)
                for step, exchange in enumerate(schedule):
                    amount = shard * len(exchange.send_slices)
                    per_rank_payload[src] += amount
                    transfers.append(_build_transfer(pname, step, src, group[exchange.send_to], amount, band, topology))
            phases.append(LoweredPhase(pname, (count - 1).bit_length(), tuple(transfers), phase_startup))
        return tuple(phases)
    if alg in {"halving_doubling", "recursive_halving_doubling"}:
        rounds = max(1, math.ceil(math.log2(count)))
        phases = []
        for pname in phase_names:
            transfers = []
            for r_idx in range(rounds):
                stride = 1 << r_idx
                if collective == "allgather":
                    chunk = api_bytes * float(stride)
                else:
                    denom = float(1 << (r_idx + 1)) if pname == "reduce_scatter" else float(1 << (rounds - r_idx))
                    chunk = api_bytes / denom
                for idx, src in enumerate(group):
                    peer_idx = idx ^ stride
                    if peer_idx >= count:
                        peer_idx = (idx + stride) % count
                    dst = group[peer_idx]
                    per_rank_payload[src] += chunk
                    transfers.append(_build_transfer(pname, r_idx, src, dst, chunk, band, topology))
            phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
        return tuple(phases)
    if alg == "tree":
        rounds = max(1, math.ceil(math.log2(count)))
        if collective == "allreduce":
            phase_names = ("reduce_up", "broadcast_down")
        elif collective == "allgather":
            phase_names = ("gather_up", "broadcast_down")
        elif collective == "reduce_scatter":
            phase_names = ("reduce_up", "scatter_down")
        else:
            raise ModelingError(f"collective_band_algorithm_unsupported:tree:{collective}")
        phases = []
        for pname in phase_names:
            transfers = []
            top_down = pname in {"broadcast_down", "scatter_down"}
            for r_idx in range(rounds):
                stride = 1 << (rounds - 1 - r_idx) if top_down else (1 << r_idx)
                for idx in range(0, count, stride * TWO_RANKS):
                    partner = idx + stride
                    if partner >= count:
                        continue
                    src, dst = (group[idx], group[partner]) if top_down else (group[partner], group[idx])
                    subtree = min(stride, count - partner)
                    if collective == "allgather":
                        chunk = (float(count) * api_bytes if top_down
                                 else float(subtree) * api_bytes)
                    elif collective == "reduce_scatter":
                        chunk = (float(subtree) * (api_bytes / float(count)) if top_down
                                 else api_bytes)
                    else:
                        chunk = api_bytes
                    per_rank_payload[src] += chunk
                    transfers.append(_build_transfer(pname, r_idx, src, dst, chunk, band, topology))
            phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
        return tuple(phases)
    if alg == "double_ring":
        pkgs = _package_partition(group, topology)
        ring_order = tuple(rank for pkg in pkgs for rank in pkg) if pkgs else group
        rounds = count - 1
        half_shard = shard / float(TWO_RANKS)
        phases = []
        for pname in phase_names:
            transfers = []
            for r_idx in range(rounds):
                for idx, src in enumerate(ring_order):
                    cw_dst = ring_order[(idx + 1) % count]
                    ccw_dst = ring_order[(idx - 1) % count]
                    per_rank_payload[src] += TWO_RANKS * half_shard
                    transfers.append(_build_transfer(pname, r_idx, src, cw_dst, half_shard, band, topology))
                    transfers.append(_build_transfer(pname, r_idx, src, ccw_dst, half_shard, band, topology))
            phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
        return tuple(phases)
    # Default: ring
    rounds = count - 1
    phases = []
    for pname in phase_names:
        transfers = []
        for r_idx in range(rounds):
            for idx, src in enumerate(group):
                dst = group[(idx + 1) % count]
                per_rank_payload[src] += shard
                transfers.append(_build_transfer(pname, r_idx, src, dst, shard, band, topology))
        phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
    return tuple(phases)


def _resolve_collective_startup_and_hbm(
    collective: str,
    group: tuple[int, ...],
    topology,
    band: BandSelection,
    phases: tuple[LoweredPhase, ...],
    api_bytes: float,
    stack: str,
    base_startup_us: float,
) -> tuple[float, bool, tuple[LoweredPhase, ...], float | None, float]:
    """Resolve calibrated collective startup, UB ceiling, and unabsorbed local HBM bytes."""
    count = len(group)
    matched_p18 = (
        float(topology.p18_matched_startup_us(collective, group, stack=stack))
        if hasattr(topology, "p18_matched_startup_us") else 0.0
    )
    matched_p19 = (
        float(topology.p19_matched_duplex_startup_us(collective, group, stack=stack))
        if hasattr(topology, "p19_matched_duplex_startup_us") else 0.0
    )
    total_rounds = max(1, sum(phase.rounds for phase in phases))
    dev_step = (
        float(topology.collective_device_step_us(
            collective, group, alpha_us=float(band.startup_us), stack=stack,
            phase_count=max(1, len(phases)), total_rounds=total_rounds,
        ))
        if hasattr(topology, "collective_device_step_us") else 0.0
    )
    collective_startup = base_startup_us
    uses_calibrated_startup = False
    if matched_p18 > 0.0:
        collective_startup = max(collective_startup, matched_p18)
        uses_calibrated_startup = True
    elif matched_p19 > 0.0:
        collective_startup = max(collective_startup, matched_p19)
        uses_calibrated_startup = True
    elif dev_step > 0.0:
        collective_startup = max(collective_startup, dev_step)
        uses_calibrated_startup = True
    if uses_calibrated_startup and phases:
        phases = tuple(replace(phase, startup_us=0.0) for phase in phases)

    fabric = band.protocol.rsplit("_", 1)[0]
    uses_per_pair_mesh = bool(
        phases
        and all(
            topology.links[hop.link_name].semantics == "per_pair"
            for phase in phases
            for transfer in phase.transfers
            for hop in transfer.hops
        )
    )
    die_inj_ceil = (
        UB_DIE_INJECTION_CEILING_BYTES_PER_MS
        if fabric == "ub_mesh" and count >= UB_INJECTION_CEILING_MIN_RANKS and uses_per_pair_mesh
        else None
    )
    has_p18_ar = bool(
        hasattr(topology, "has_calibrated_p18_allreduce_rate")
        and topology.has_calibrated_p18_allreduce_rate()
    )
    local_hbm_bytes = 0.0
    if api_bytes > 0.0 and count >= TWO_RANKS and uses_calibrated_startup:
        if collective in {"reduce_scatter", "allreduce"} and has_p18_ar:
            # Ring/mesh ReduceScatter executes (W-1) binary adds on N/W shards = 3*(W-1)/W * N HBM.
            # Subtract the 1.5 HBM/wire already embedded in 2-rank P18 AllReduce on the bottleneck fabric.
            total_reduce_hbm = float(LOCAL_REDUCE_HBM_PASSES) * float(count - 1) / float(count) * api_bytes
            bottleneck_wire = float(count - 1) / float(count) * api_bytes
            if collective == "allreduce":
                bottleneck_wire *= float(TWO_RANKS)
            if band.algorithm == "double_ring":
                bottleneck_wire /= float(TWO_RANKS)
            local_hbm_bytes = max(
                0.0,
                total_reduce_hbm - P18_ALLREDUCE_EMBEDDED_HBM_PER_WIRE * bottleneck_wire,
            )
        elif (
            collective == "alltoall"
            and fabric == "hccs_k8"
            and api_bytes <= float(HCCL_DEFAULT_BUFFSIZE_BYTES)
        ):
            local_hbm_bytes = float(SCRATCH_STAGING_HBM_PASSES) * api_bytes
    return collective_startup, uses_calibrated_startup, phases, die_inj_ceil, local_hbm_bytes


def lower_collective(
    call,
    actual_group: Sequence[int],
    topology,
    api_bytes_per_rank: int | float,
    *,
    stack: str = "default",
    expansion_mode: str | None = None,
    peer_matrix: Sequence[Sequence[int]] | None = None,
    send_splits: Sequence[int] | None = None,
    recv_splits: Sequence[int] | None = None,
    communicator_id: str = "",
    coset_id: str = "",
) -> LoweredCollective:
    """Lower a collective call into framed physical phases, hops, and port keys."""
    collective = _normalize_collective_name(call)
    group = _normalize_group(actual_group)
    count = len(group)
    if (isinstance(api_bytes_per_rank, bool)
            or not isinstance(api_bytes_per_rank, (int, float))
            or not math.isfinite(float(api_bytes_per_rank))
            or float(api_bytes_per_rank) < 0):
        raise ModelingError(f"collective_api_bytes_invalid:{api_bytes_per_rank}")
    api_bytes = float(api_bytes_per_rank)

    matrix: tuple[tuple[int, ...], ...] | None = None
    band_payload = api_bytes
    if collective == "alltoallv":
        matrix = _validate_alltoallv_inputs(
            peer_matrix, count, api_bytes, send_splits, recv_splits
        )
        row_sums = tuple(sum(row) for row in matrix)
        band_payload = float(max(row_sums)) if sum(row_sums) > 0 else 0.0

    band = select_band(
        collective, group, topology, band_payload,
        stack="default" if stack == "eager" else stack, expansion_mode=expansion_mode,
    )
    collective_startup = band.startup_us if band.startup_scope == "collective" else 0.0
    phase_startup = band.startup_us if band.startup_scope == "phase" else 0.0
    self_copies: dict[int, float] = {rank: 0.0 for rank in group}
    per_rank_payload: dict[int, float] = {rank: 0.0 for rank in group}

    if collective == "alltoallv":
        assert matrix is not None
        transfers: list[LoweredTransfer] = []
        for s_idx, src in enumerate(group):
            self_copies[src] = float(matrix[s_idx][s_idx])
            for d_idx, dst in enumerate(group):
                if s_idx == d_idx or matrix[s_idx][d_idx] == 0:
                    continue
                amount = float(matrix[s_idx][d_idx])
                per_rank_payload[src] += amount
                transfers.append(_build_transfer("alltoallv", 0, src, dst, amount, band, topology))
        phases = (LoweredPhase("alltoallv", 1, tuple(transfers), phase_startup),) if transfers else ()
    elif api_bytes == 0.0 or count <= 1:
        phases = ()
    elif collective in {"allreduce", "reduce_scatter", "allgather"}:
        phases = _lower_ring_or_alg(
            collective, group, api_bytes, band, topology, phase_startup, per_rank_payload
        )
        if collective == "allreduce" and (
            band.algorithm in _MESH_ALGORITHMS
            or band.algorithm in {"ring", "double_ring", "nhr", "hierarchical_2d"}
        ):
            one_shot = band.algorithm in {"one_shot_mesh", "small_count_mesh"}
            expected = (count - 1) * api_bytes * (1 if one_shot else 2 / count)
            for rank in group:
                if not math.isclose(per_rank_payload[rank], expected, rel_tol=1e-12, abs_tol=1e-9):
                    raise ModelingError(f"ring_allreduce_payload_conservation_failed:{rank}")
    else:
        # Balanced AllToAll: total send buffer B splits B/P per peer (including self)
        per_peer = api_bytes / float(count)
        transfers = []
        concurrent = band.algorithm in {"direct", "direct_mesh", "pairwise"}
        for s_idx, src in enumerate(group):
            self_copies[src] = per_peer
            for step in range(count - 1):
                dst = group[(s_idx + step + 1) % count]
                per_rank_payload[src] += per_peer
                r_idx = 0 if concurrent else step
                transfers.append(_build_transfer("alltoall", r_idx, src, dst, per_peer, band, topology))
        rounds = 1 if concurrent else (count - 1)
        phases = (LoweredPhase("alltoall", rounds, tuple(transfers), phase_startup),)

    (
        collective_startup,
        uses_calibrated_startup,
        phases,
        die_inj_ceil,
        local_hbm_bytes,
    ) = _resolve_collective_startup_and_hbm(
        collective, group, topology, band, phases, api_bytes, stack, collective_startup
    )
    total_payload = sum(per_rank_payload.values())
    total_wire = sum(t.wire_bytes for phase in phases for t in phase.transfers)
    return LoweredCollective(
        collective=collective, group=group, band=band,
        communicator_id=str(communicator_id), coset_id=str(coset_id),
        api_bytes_per_rank=api_bytes, rendezvous_required=True,
        phases=phases, total_payload_bytes=total_payload, total_wire_bytes=total_wire,
        per_rank_payload_bytes=tuple((r, per_rank_payload[r]) for r in group),
        self_copy_bytes_by_rank=tuple((r, self_copies[r]) for r in group),
        collective_startup_us=collective_startup,
        uses_calibrated_startup=uses_calibrated_startup,
        die_injection_ceiling_bytes_per_ms=die_inj_ceil,
        local_hbm_bytes_per_rank=local_hbm_bytes,
    )


def _submit_transfer_chunks(
    engine,
    prefix: str,
    transfer: LoweredTransfer,
    gate,
    *,
    band: CollectiveBandSpec,
    credit_cap: float,
    pool_name: str | None,
    hop_startup_ms: float,
    uses_calibrated_startup: bool,
    ub_tx_res: Mapping[int, str],
    ub_rx_res: Mapping[int, str],
    flow_tx_res: Mapping[int, str],
    flow_rx_res: Mapping[int, str],
    owner: str,
) -> list:
    pipelined_credit = bool(
        uses_calibrated_startup
        and credit_cap > 0.0
        and credit_cap >= float(DOUBLE_BUFFER_CREDIT_FRAMES * band.wire_bytes_per_frame)
    )
    if credit_cap > 0.0 and transfer.wire_bytes > credit_cap:
        slice_cap = credit_cap / float(DOUBLE_BUFFER_CREDIT_FRAMES) if pipelined_credit else credit_cap
        chunks = chunk_sizes(transfer.wire_bytes, slice_cap, DEFAULT_MAX_PHYSICAL_CHUNKS)
    else:
        chunks = (transfer.wire_bytes,)
    done = []
    for c_idx, chunk_wire in enumerate(chunks):
        alloc = (
            Allocation(f"{prefix}:c{c_idx}:cr", pool_name, min(chunk_wire, credit_cap), (0,))
            if pool_name is not None and chunk_wire > 0.0
            else None
        )
        hop_cursor = gate
        for h_idx, hop in enumerate(transfer.hops):
            demand = {res: chunk_wire for res in hop.resources}
            tx_inj = ub_tx_res.get(hop.source)
            rx_inj = ub_rx_res.get(hop.destination)
            if tx_inj is not None:
                demand[tx_inj] = chunk_wire
            if rx_inj is not None:
                demand[rx_inj] = chunk_wire
            if transfer.flow_cap_bytes_per_ms is not None:
                sdma_tx = flow_tx_res.get(hop.source)
                sdma_rx = flow_rx_res.get(hop.destination)
                if sdma_tx is not None:
                    demand[sdma_tx] = chunk_wire
                if sdma_rx is not None:
                    demand[sdma_rx] = chunk_wire
            svc = engine.submit(
                f"{prefix}:c{c_idx}:h{h_idx}:svc",
                demand,
                after=(hop_cursor,),
                acquire=(alloc,) if (h_idx == 0 and alloc is not None) else (),
                owner=owner,
                coupled=len(demand) > 1,
            )
            is_last = h_idx == len(transfer.hops) - 1
            if uses_calibrated_startup:
                hop_lat_ms = hop.switch_propagation_ms
            elif band.startup_us > 0.0:
                charge_startup = not pipelined_credit or c_idx == 0
                hop_lat_ms = (hop_startup_ms if charge_startup else 0.0) + hop.switch_propagation_ms
            else:
                hop_lat_ms = hop.propagation_ms
            hop_cursor = engine.submit(
                f"{prefix}:c{c_idx}:h{h_idx}:arr",
                after=(svc,),
                latency=hop_lat_ms,
                free=(alloc,) if (is_last and alloc is not None) else (),
                owner=owner,
            )
        done.append(hop_cursor)
    return done


def _prepare_collective_resources(
    engine, name: str, lowered: LoweredCollective, credit_cap: float
) -> tuple[dict[tuple[int, int] | int, str], dict[int, str], dict[int, str], dict[int, str], dict[int, str]]:
    credit_pools: dict[tuple[int, int] | int, str] = {}
    new_pools: dict[str, float] = {}
    flow_tx_res: dict[int, str] = {}
    flow_rx_res: dict[int, str] = {}
    new_resources: list[Resource] = []
    for phase in lowered.phases:
        for transfer in phase.transfers:
            if credit_cap > 0.0:
                if lowered.uses_calibrated_startup:
                    key: tuple[int, int] | int = (transfer.source, transfer.destination)
                    pool_name = f"{name}:credit:r{transfer.source}to{transfer.destination}"
                else:
                    key = transfer.destination
                    pool_name = f"{name}:credit:r{transfer.destination}"
                if key not in credit_pools:
                    credit_pools[key] = pool_name
                    if pool_name not in engine.pools:
                        new_pools[pool_name] = credit_cap
            if transfer.flow_cap_bytes_per_ms is not None:
                cap = float(transfer.flow_cap_bytes_per_ms)
                for hop in transfer.hops:
                    if hop.source not in flow_tx_res:
                        tx_name = f"{name}:sdma_flow:tx:r{hop.source}"
                        flow_tx_res[hop.source] = tx_name
                        if tx_name not in engine.resources:
                            new_resources.append(Resource(tx_name, cap, "bytes"))
                    if hop.destination not in flow_rx_res:
                        rx_name = f"{name}:sdma_flow:rx:r{hop.destination}"
                        flow_rx_res[hop.destination] = rx_name
                        if rx_name not in engine.resources:
                            new_resources.append(Resource(rx_name, cap, "bytes"))
    ub_tx_res: dict[int, str] = {}
    ub_rx_res: dict[int, str] = {}
    if lowered.die_injection_ceiling_bytes_per_ms is not None:
        ceil = float(lowered.die_injection_ceiling_bytes_per_ms)
        for rank in lowered.group:
            tx_name = f"{name}:ub_inj:tx:r{rank}"
            rx_name = f"{name}:ub_inj:rx:r{rank}"
            ub_tx_res[rank] = tx_name
            ub_rx_res[rank] = rx_name
            if tx_name not in engine.resources:
                new_resources.append(Resource(tx_name, ceil, "bytes"))
            if rx_name not in engine.resources:
                new_resources.append(Resource(rx_name, ceil, "bytes"))
    if new_resources or new_pools:
        engine.extend(tuple(new_resources), pools=new_pools or None)
    return credit_pools, ub_tx_res, ub_rx_res, flow_tx_res, flow_rx_res


def submit_lowered_collective(
    engine,
    name: str,
    lowered: LoweredCollective,
    ready_by_rank: Mapping[int, object],
    *,
    owner: str = "",
):
    """Submit a ``LoweredCollective`` onto ``engine`` with per-hop & credit physics."""
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
        0.0
        if lowered.uses_calibrated_startup
        else (lowered.band.startup_us / MS_PER_S if lowered.band.startup_scope == "hop" else 0.0)
    )
    credit_cap = float(lowered.band.credit_bytes or 0)
    credit_pools, ub_tx_res, ub_rx_res, flow_tx_res, flow_rx_res = _prepare_collective_resources(
        engine, name, lowered, credit_cap
    )

    if all(phase.pipelined_slices for phase in lowered.phases):
        slice_ids = sorted({t.slice_index for phase in lowered.phases for t in phase.transfers})
        slice_cursor = {s_idx: rendezvous for s_idx in slice_ids}
        for p_idx, phase in enumerate(lowered.phases):
            by_slice: dict[int, list[LoweredTransfer]] = {}
            for transfer in phase.transfers:
                by_slice.setdefault(transfer.slice_index, []).append(transfer)
            for s_idx in slice_ids:
                s_transfers = by_slice.get(s_idx)
                if not s_transfers:
                    continue
                s_gate = (
                    engine.submit(
                        f"{name}:p{p_idx}:{phase.name}:s{s_idx}:start",
                        after=(slice_cursor[s_idx],),
                        latency=phase.startup_us / MS_PER_S,
                        owner=owner,
                    )
                    if (s_idx == slice_ids[0] and phase.startup_us > 0.0)
                    else slice_cursor[s_idx]
                )
                s_done = []
                for t_idx, transfer in enumerate(s_transfers):
                    pool_key = (
                        (transfer.source, transfer.destination)
                        if lowered.uses_calibrated_startup else transfer.destination
                    )
                    s_done.extend(_submit_transfer_chunks(
                        engine, f"{name}:p{p_idx}:s{s_idx}:t{t_idx}", transfer, s_gate,
                        band=lowered.band, credit_cap=credit_cap,
                        pool_name=credit_pools.get(pool_key),
                        hop_startup_ms=hop_startup_ms,
                        uses_calibrated_startup=lowered.uses_calibrated_startup,
                        ub_tx_res=ub_tx_res, ub_rx_res=ub_rx_res,
                        flow_tx_res=flow_tx_res, flow_rx_res=flow_rx_res, owner=owner,
                    ))
                slice_cursor[s_idx] = engine.submit(
                    f"{name}:p{p_idx}:s{s_idx}:done",
                    after=tuple(s_done),
                    owner=owner,
                )
        complete = engine.submit(f"{name}:complete", after=tuple(slice_cursor.values()), owner=owner)
        return {rank: complete for rank in lowered.group}

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
                pool_key = (
                    (transfer.source, transfer.destination)
                    if lowered.uses_calibrated_startup else transfer.destination
                )
                round_done.extend(_submit_transfer_chunks(
                    engine, f"{name}:p{p_idx}:r{round_index}:t{t_idx}", transfer, round_gate,
                    band=lowered.band, credit_cap=credit_cap,
                    pool_name=credit_pools.get(pool_key),
                    hop_startup_ms=hop_startup_ms,
                    uses_calibrated_startup=lowered.uses_calibrated_startup,
                    ub_tx_res=ub_tx_res, ub_rx_res=ub_rx_res,
                    flow_tx_res=flow_tx_res, flow_rx_res=flow_rx_res, owner=owner,
                ))
            round_gate = engine.submit(
                f"{name}:p{p_idx}:r{round_index}:done",
                after=tuple(round_done),
                owner=owner,
            )
        cursor = round_gate

    complete = engine.submit(f"{name}:complete", after=(cursor,), owner=owner)
    return {rank: complete for rank in lowered.group}
