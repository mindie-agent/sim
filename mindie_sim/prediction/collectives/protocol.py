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

from mindie_sim.backends.ascend import A5_POD_RANKS_PER_NODE
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.hardware_links import CollectiveBandSpec
from ..engine import Allocation, Resource
from mindie_sim.prediction.operators.collectives.hccl_nhr import rounds as nhr_rounds
from .ports import (
    PairKey,
    PortKey,
    die_injection_resource_name,
    link_pair_key,
    link_port_key,
    sdma_flow_resource_name,
)
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
DOUBLE_BUFFER_CREDIT_FRAMES = 2
_HCOMM_EXPANSIONS = {
    "aiv": "aiv", "ccu_ms": "ccu_ms", "ccu_sched": "ccu_sched",
    "ai_cpu": "aicpu", "aicpu": "aicpu", "aicpu_ts": "aicpu", "nhr": "nhr",
}
_PROCESS_ADVISORY_EXPANSIONS = frozenset({"aiv", "ccu_ms", "ccu_sched"})
# UB mesh spans the A5 Pod node scope (ascend.json group_max=64); a5_server ranks_per_node=8 still bounds it.
_PHYSICAL_BAND_LIMITS = {"hccs_k8": K8_RANKS, "ub_mesh": A5_POD_RANKS_PER_NODE, "hccs_sw": A3_NODE_RANKS, "sio": TWO_RANKS}
_MESH_ALGORITHMS = frozenset({
    "direct", "direct_mesh", "pairwise", "mesh", "small_count_mesh", "one_shot_mesh",
    "two_shot_mesh", "chunked_two_shot_mesh", "chunked_mesh",
})
_CONCURRENT_MESH_ALGORITHMS = _MESH_ALGORITHMS - frozenset({"pairwise"})
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
    flow_cap_bytes_per_ms: float | None = None


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
    lane_index: int = 0

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
    namespace: str = ""

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
            if collective == "allreduce":
                return "one_shot_mesh"
            if collective == "reduce_scatter":
                return "mesh"
        if collective in {"alltoall", "alltoallv"} and (
            expansion is None or (fabric == "hccs_optical" and expansion == "aiv")
        ):
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
        if fabric == "hccs_optical" and collective in {"allgather", "reduce_scatter"} and expansion == "aiv":
            return "mesh"
    if fabric in {"ub_mesh", "ub_superpod"}:
        if collective == "allreduce":
            if expansion in (None, "ccu_ms"):
                return "one_shot_mesh" if size <= ranks * A5_CCU_ONE_SHOT_SHARD_BYTES else "mesh"
            if expansion == "ccu_sched":
                return "mesh"
            if expansion == "aicpu":
                if size <= A5_AICPU_ONE_SHOT_BYTES:
                    return "one_shot_mesh"
                if size * K8_RANKS > A5_AICPU_ALLREDUCE_CHUNK_BYTES * ranks ** 2:
                    return "chunked_two_shot_mesh"
                return "two_shot_mesh"
        if collective in {"allgather", "reduce_scatter"} and expansion in (None, "ccu_ms", "ccu_sched"):
            return "mesh"
        if collective in {"alltoall", "alltoallv"} and expansion in (None, "ccu_ms", "ccu_sched"):
            return "direct_mesh"
        if collective == "reduce_scatter" and expansion == "aicpu":
            return ("chunked_mesh" if size * K8_RANKS ** 2 > A5_AICPU_RS_CHUNK_BYTES * ranks ** 2
                    else "mesh")
    return None


def _match_candidate_bands(
    candidates: tuple[CollectiveBandSpec, ...],
    *,
    collective: str,
    group_len: int,
    payload: float,
    stack_key: str,
    scope: str,
    exp_key: str | None,
    is_multi_pkg: bool,
) -> tuple[list[tuple[tuple[int, int, int, int], CollectiveBandSpec]], int | None]:
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
        if band.algorithm == "ring" and (exp_key in (None, "concurrent") or exp_key in _HCOMM_EXPANSIONS):
            spec_exp = None if exp_key == "concurrent" else _HCOMM_EXPANSIONS.get(exp_key)
            algorithm = _hcomm_algorithm(
                fabric, collective, group_len, payload, spec_exp,
                is_multi_package=is_multi_pkg,
            )
        explicit_executor = exp_key in _HCOMM_EXPANSIONS and algorithm is not None
        band_exp_matches = (
            exp_key is None
            or band.expansion_mode in (exp_key, "*")
            or explicit_executor
        )
        if not band_exp_matches:
            continue
        limit = _PHYSICAL_BAND_LIMITS.get(fabric)
        if limit is not None and group_len > limit:
            physical_limit = limit
            continue
        group_max = min(band.group_max, limit) if limit is not None else band.group_max
        if not band.group_min <= group_len <= group_max:
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
    return matches, physical_limit


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
    is_multi_pkg = bool(_package_partition(group, topology))

    matches, physical_limit = _match_candidate_bands(
        candidates, collective=collective, group_len=len(group), payload=payload,
        stack_key=stack_key, scope=scope, exp_key=exp_key, is_multi_pkg=is_multi_pkg,
    )
    if not matches and exp_key in _PROCESS_ADVISORY_EXPANSIONS:
        advisory_candidates = tuple(
            band for band in candidates if band.expansion_mode not in _PROCESS_ADVISORY_EXPANSIONS
        )
        matches, physical_limit = _match_candidate_bands(
            advisory_candidates, collective=collective, group_len=len(group), payload=payload,
            stack_key=stack_key, scope=scope, exp_key=None, is_multi_pkg=is_multi_pkg,
        )

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
) -> tuple[int, int | float]:
    """Compute ``(frame_count, wire_bytes)`` under ``band`` MTU + framing policy."""
    is_exact_int = type(payload_bytes) is int
    if is_exact_int:
        if payload_bytes <= 0:
            return 0, 0
        exact: int | None = payload_bytes
        payload = float(payload_bytes)
    else:
        payload = float(payload_bytes)
        if payload <= 0.0:
            return 0, 0.0
        exact = int(payload) if payload.is_integer() else None
    pbf = int(band.payload_bytes_per_frame)
    wbf = int(band.wire_bytes_per_frame)
    if exact is not None:
        frames = (exact + pbf - 1) // pbf
    else:
        frames = max(1, math.ceil(payload / float(pbf) - 1e-12))
    policy = getattr(band, "short_final_frame_policy", "exact_plus_header")
    if policy == "pad_full_frame":
        total = frames * wbf
        return frames, total if is_exact_int else float(total)
    if policy != "exact_plus_header":
        raise ModelingError(f"collective_band_short_final_frame_policy_invalid:{policy}")
    header_per_frame = max(0, wbf - pbf)
    if exact is not None:
        total = exact + frames * header_per_frame
        return frames, total if is_exact_int else float(total)
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


def _resolve_element_bytes(call, dtype: str | None = None) -> int:
    """Resolve positive integer element width for shard alignment (default 1 B)."""
    raw_dtype = dtype
    if not raw_dtype:
        if isinstance(call, Mapping):
            raw_dtype = call.get("dtype")
        elif not isinstance(call, str):
            raw_dtype = getattr(call, "dtype", None)
            if not raw_dtype:
                meta = dict(getattr(call, "metadata", ()) or ())
                raw_dtype = meta.get("dtype") or meta.get("payload_dtype")
    if not raw_dtype:
        return 1
    try:
        width = float(dtype_bytes(str(raw_dtype)))
    except ModelingError:
        return 1
    return int(width) if width >= 1.0 and width.is_integer() else 1


def _build_transfer(
    phase_name: str, round_index: int, source: int, destination: int,
    payload_bytes: int | float, band: CollectiveBandSpec, topology,
    *, slice_index: int = 0, lane_index: int = 0,
) -> LoweredTransfer:
    frames, wire_bytes = frame_wire_bytes(payload_bytes, band)
    path = topology.route(source, destination)
    hops: list[LoweredHop] = []
    flow_caps: list[float] = []
    cap_fn = getattr(topology, "single_peer_flow_cap_bytes_per_ms", None)
    for link_name, hop_src, hop_dst in path:
        link = topology.links[link_name]
        hop_cap: float | None = None
        if cap_fn is not None:
            cap_val = cap_fn(link_name)
            if cap_val is not None:
                hop_cap = float(cap_val)
                flow_caps.append(hop_cap)
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
            flow_cap_bytes_per_ms=hop_cap,
        ))
    return LoweredTransfer(
        phase_name=phase_name, round_index=round_index,
        source=int(source), destination=int(destination),
        payload_bytes=float(payload_bytes), frames=frames,
        wire_bytes=float(wire_bytes), hops=tuple(hops),
        slice_index=int(slice_index),
        flow_cap_bytes_per_ms=min(flow_caps) if flow_caps else None,
        lane_index=int(lane_index),
    )


def _subgroup_mesh_transfers(
    phase_name: str, subgroups: Sequence[tuple[int, ...]],
    shards_by_group: Sequence[tuple[float, ...]],
    band: CollectiveBandSpec, topology, per_rank_payload: dict[int, float],
    *, slice_index: int, by_destination: bool,
) -> list[LoweredTransfer]:
    out: list[LoweredTransfer] = []
    for sub, sub_shards in zip(subgroups, shards_by_group):
        for s_idx, src in enumerate(sub):
            for d_idx, dst in enumerate(sub):
                if src == dst:
                    continue
                amount = sub_shards[d_idx if by_destination else s_idx]
                if amount <= 0.0:
                    continue
                per_rank_payload[src] += amount
                out.append(_build_transfer(
                    phase_name, 0, src, dst, amount, band, topology,
                    slice_index=slice_index,
                ))
    return out


def _lower_hierarchical_2d(
    collective: str, group: tuple[int, ...], api_bytes: float,
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
    *, element_bytes: int = 1,
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
    snapped_api = _snap_integral_bytes(api_bytes)
    elem_q = (
        int(element_bytes)
        if element_bytes > 1 and snapped_api.is_integer() and int(snapped_api) % int(element_bytes) == 0
        else 1
    )
    group_q = len(group) * elem_q
    slice_q = (
        group_q
        if collective != "allgather" and snapped_api.is_integer() and int(snapped_api) % group_q == 0
        else elem_q
    )
    slice_apis = _integer_shards(snapped_api, n_slices, quantum=slice_q)
    if collective == "allreduce":
        stage_names = ("intra_rs", "inter_rs", "inter_ag", "intra_ag")
    elif collective == "allgather":
        stage_names = ("intra_ag", "inter_ag")
    else:
        stage_names = ("inter_rs", "intra_rs")
    stage_transfers: dict[str, list[LoweredTransfer]] = {name: [] for name in stage_names}
    for s_idx, s_api in enumerate(slice_apis):
        if s_api <= 0.0:
            continue
        if collective == "allreduce":
            die_shards = _integer_shards(s_api, n_die, quantum=elem_q)
            leaf_by_slot = tuple(
                _integer_shards(die_shards[slot], n_pkg, quantum=elem_q)
                for slot in range(n_die)
            )
            specs = (
                ("intra_rs", pkgs, (die_shards,) * n_pkg, True),
                ("inter_rs", inter_groups, leaf_by_slot, True),
                ("inter_ag", inter_groups, leaf_by_slot, False),
                ("intra_ag", pkgs, (die_shards,) * n_pkg, False),
            )
        elif collective == "allgather":
            specs = (
                ("intra_ag", pkgs, ((s_api,) * n_die,) * n_pkg, False),
                ("inter_ag", inter_groups, ((s_api * float(n_die),) * n_pkg,) * n_die, False),
            )
        else:
            pkg_shards = _integer_shards(s_api, n_pkg, quantum=elem_q)
            leaf_by_pkg = tuple(
                _integer_shards(pkg_shards[p], n_die, quantum=elem_q)
                for p in range(n_pkg)
            )
            specs = (
                ("inter_rs", inter_groups, (pkg_shards,) * n_die, True),
                ("intra_rs", pkgs, leaf_by_pkg, True),
            )
        for s_name, subgroups, group_shards, by_dst in specs:
            stage_transfers[s_name].extend(_subgroup_mesh_transfers(
                s_name, subgroups, group_shards, band, topology, per_rank_payload,
                slice_index=s_idx, by_destination=by_dst,
            ))
    return tuple(
        LoweredPhase(
            s_name, 1, tuple(stage_transfers[s_name]), phase_startup,
            pipelined_slices=n_slices > 1,
        )
        for s_name in stage_names
    )


def _snap_integral_bytes(value: float) -> float:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-9):
        return float(nearest)
    return float(value)


def _integer_shards(total_bytes: float, parts: int, *, quantum: int = 1) -> tuple[float, ...]:
    """Partition ``total_bytes`` into exact integer ``quantum``-byte shards when integral."""
    snapped = _snap_integral_bytes(total_bytes)
    if snapped.is_integer():
        total_int = int(snapped)
        step = int(quantum) if quantum > 1 and total_int % int(quantum) == 0 else 1
        base, rem = divmod(total_int // step, parts)
        return tuple(float((base + (1 if idx < rem else 0)) * step) for idx in range(parts))
    even = total_bytes / float(parts)
    return (even,) * parts


def _lower_mesh(
    collective: str, group: tuple[int, ...], api_bytes: float,
    shards: tuple[float, ...], alg: str, phase_names: tuple[str, ...],
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
    *, element_bytes: int = 1,
) -> tuple[LoweredPhase, ...]:
    count = len(group)
    single_exchange = collective == "allreduce" and (
        alg in {"one_shot_mesh", "small_count_mesh"} or (alg == "direct" and count == TWO_RANKS)
    )
    if single_exchange:
        phase_names = ("direct" if alg == "direct" else "all_exchange",)
    phases = []
    for pname in phase_names:
        rounds = count - 1 if pname == "reduce_scatter" and alg.startswith("chunked_") else 1
        transfers: list[LoweredTransfer] = []
        for s_idx, src in enumerate(group):
            for d_idx, dst in enumerate(group):
                if src == dst:
                    continue
                if single_exchange or collective == "allgather":
                    total_amt = api_bytes
                elif pname == "reduce_scatter":
                    total_amt = shards[d_idx]
                else:
                    total_amt = shards[s_idx]
                if total_amt <= 0.0:
                    continue
                step_amounts = (
                    _integer_shards(total_amt, rounds, quantum=element_bytes)
                    if rounds > 1 else (total_amt,)
                )
                for step, amount in enumerate(step_amounts):
                    if amount <= 0.0:
                        continue
                    per_rank_payload[src] += amount
                    transfers.append(_build_transfer(pname, step, src, dst, amount, band, topology))
        phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
    return tuple(phases)


def _lower_halving_doubling(
    collective: str, group: tuple[int, ...], api_bytes: float,
    shards: tuple[float, ...], phase_names: tuple[str, ...],
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
) -> tuple[LoweredPhase, ...]:
    count = len(group)
    if count & (count - 1):
        raise ModelingError(f"collective_halving_doubling_requires_power_of_two_ranks:{count}")
    rounds = max(1, (count - 1).bit_length())
    phases = []
    for pname in phase_names:
        transfers: list[LoweredTransfer] = []
        for r_idx in range(rounds):
            stride = 1 << r_idx
            rs_mask = (1 << (r_idx + 1)) - 1
            for idx, src in enumerate(group):
                peer_idx = idx ^ stride
                dst = group[peer_idx]
                if collective == "allgather":
                    chunk = api_bytes * float(stride)
                elif pname == "reduce_scatter":
                    target_bits = peer_idx & rs_mask
                    chunk = math.fsum(
                        shards[k] for k in range(count) if (k & rs_mask) == target_bits
                    )
                else:
                    block_bits = idx >> r_idx
                    chunk = math.fsum(
                        shards[k] for k in range(count) if (k >> r_idx) == block_bits
                    )
                if chunk <= 0.0:
                    continue
                per_rank_payload[src] += chunk
                transfers.append(_build_transfer(pname, r_idx, src, dst, chunk, band, topology))
        phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
    return tuple(phases)


def _lower_tree(
    collective: str, group: tuple[int, ...], api_bytes: float,
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
) -> tuple[LoweredPhase, ...]:
    count = len(group)
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


def _lower_ring(
    collective: str, group: tuple[int, ...], api_bytes: float,
    shards: tuple[float, ...], phase_names: tuple[str, ...],
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
) -> tuple[LoweredPhase, ...]:
    count = len(group)
    rounds = count - 1
    phases = []
    for pname in phase_names:
        transfers: list[LoweredTransfer] = []
        for r_idx in range(rounds):
            for idx, src in enumerate(group):
                dst = group[(idx + 1) % count]
                if collective == "allgather":
                    amount = api_bytes
                elif pname == "reduce_scatter":
                    amount = shards[(idx - r_idx) % count]
                else:
                    amount = shards[(idx + 1 - r_idx) % count]
                if amount <= 0.0:
                    continue
                per_rank_payload[src] += amount
                transfers.append(_build_transfer(pname, r_idx, src, dst, amount, band, topology))
        phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
    return tuple(phases)


def _lower_ring_or_alg(
    collective: str, group: tuple[int, ...], api_bytes: float,
    band: CollectiveBandSpec, topology, phase_startup: float,
    per_rank_payload: dict[int, float],
    *, element_bytes: int = 1,
) -> tuple[LoweredPhase, ...]:
    count = len(group)
    alg = str(band.algorithm).strip().lower()
    if alg not in SUPPORTED_BAND_ALGORITHMS:
        raise ModelingError(f"collective_band_algorithm_unsupported:{band.algorithm}")
    phase_names = ("reduce_scatter", "all_gather") if collective == "allreduce" else (collective,)
    shards = (
        (api_bytes,) * count
        if collective == "allgather"
        else _integer_shards(api_bytes, count, quantum=element_bytes)
    )
    if alg == "hierarchical_2d":
        hier_phases = _lower_hierarchical_2d(
            collective, group, api_bytes, band, topology, phase_startup, per_rank_payload,
            element_bytes=element_bytes,
        )
        if hier_phases:
            return hier_phases
    if alg in _MESH_ALGORITHMS:
        return _lower_mesh(
            collective, group, api_bytes, shards, alg, phase_names,
            band, topology, phase_startup, per_rank_payload,
            element_bytes=element_bytes,
        )
    if alg == "nhr":
        if count & (count - 1):
            raise ModelingError("collective_nhr_requires_power_of_two_ranks")
        phases = []
        for pname in phase_names:
            transfers = []
            for index, src in enumerate(group):
                for step, exchange in enumerate(nhr_rounds(count, index, pname)):
                    amount = math.fsum(shards[s] for s in exchange.send_slices)
                    if amount <= 0.0:
                        continue
                    per_rank_payload[src] += amount
                    transfers.append(_build_transfer(pname, step, src, group[exchange.send_to], amount, band, topology))
            phases.append(LoweredPhase(pname, (count - 1).bit_length(), tuple(transfers), phase_startup))
        return tuple(phases)
    if alg in {"halving_doubling", "recursive_halving_doubling"}:
        return _lower_halving_doubling(
            collective, group, api_bytes, shards, phase_names,
            band, topology, phase_startup, per_rank_payload,
        )
    if alg == "tree":
        return _lower_tree(
            collective, group, api_bytes, band, topology, phase_startup, per_rank_payload,
        )
    if alg == "double_ring":
        pkgs = _package_partition(group, topology)
        ring_order = tuple(rank for pkg in pkgs for rank in pkg) if pkgs else group
        rounds = count - 1
        split_shards = tuple(_integer_shards(s, TWO_RANKS, quantum=element_bytes) for s in shards)
        cw_shards = tuple(pair[0] for pair in split_shards)
        ccw_shards = tuple(pair[1] for pair in split_shards)
        phases = []
        for pname in phase_names:
            transfers = []
            for r_idx in range(rounds):
                for idx, src in enumerate(ring_order):
                    cw_dst = ring_order[(idx + 1) % count]
                    ccw_dst = ring_order[(idx - 1) % count]
                    if collective == "allgather":
                        cw_amt, ccw_amt = cw_shards[idx], ccw_shards[idx]
                    elif pname == "reduce_scatter":
                        cw_amt = cw_shards[(idx - r_idx) % count]
                        ccw_amt = ccw_shards[(idx + r_idx) % count]
                    else:
                        cw_amt = cw_shards[(idx + 1 - r_idx) % count]
                        ccw_amt = ccw_shards[(idx - 1 + r_idx) % count]
                    if cw_amt > 0.0:
                        per_rank_payload[src] += cw_amt
                        transfers.append(_build_transfer(
                            pname, r_idx, src, cw_dst, cw_amt, band, topology, lane_index=0,
                        ))
                    if ccw_amt > 0.0:
                        per_rank_payload[src] += ccw_amt
                        transfers.append(_build_transfer(
                            pname, r_idx, src, ccw_dst, ccw_amt, band, topology, lane_index=1,
                        ))
            phases.append(LoweredPhase(pname, rounds, tuple(transfers), phase_startup))
        return tuple(phases)
    return _lower_ring(
        collective, group, api_bytes, shards, phase_names,
        band, topology, phase_startup, per_rank_payload,
    )


def _resolve_collective_startup_and_hbm(
    collective: str,
    group: tuple[int, ...],
    topology,
    band: CollectiveBandSpec,
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
            startup_scope=str(getattr(band, "startup_scope", "step") or "step"),
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
        if fabric == "ub_mesh" and uses_per_pair_mesh
        else None
    )
    scope = _infer_group_scope(group, topology)
    if hasattr(topology, "p18_allreduce_rate_applies"):
        has_p18_ar = bool(
            topology.p18_allreduce_rate_applies(
                collective, group, algorithm=str(band.algorithm), scope=scope,
            )
        )
    else:
        has_p18_ar = bool(
            hasattr(topology, "has_calibrated_p18_allreduce_rate")
            and topology.has_calibrated_p18_allreduce_rate()
        )
    local_hbm_bytes = 0.0
    if api_bytes > 0.0 and count >= TWO_RANKS:
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


def _validate_allreduce_conservation(
    group: tuple[int, ...],
    api_bytes: float,
    band: CollectiveBandSpec,
    per_rank_payload: Mapping[int, float],
    element_bytes: int,
) -> None:
    if not (
        band.algorithm in _MESH_ALGORITHMS
        or band.algorithm in {"ring", "double_ring", "nhr", "hierarchical_2d"}
    ):
        return
    count = len(group)
    one_shot = band.algorithm in {"one_shot_mesh", "small_count_mesh"}
    expected_total = count * (count - 1) * api_bytes * (1 if one_shot else 2 / count)
    if not math.isclose(
        math.fsum(per_rank_payload.values()), expected_total, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ModelingError("ring_allreduce_payload_conservation_failed:group")
    snapped_api = _snap_integral_bytes(api_bytes)
    eff_elem = (
        element_bytes
        if element_bytes > 1 and snapped_api.is_integer() and int(snapped_api) % element_bytes == 0
        else 1
    )
    if not (one_shot or not snapped_api.is_integer() or int(snapped_api) % (count * eff_elem) == 0):
        return
    expected = expected_total / count
    for rank in group:
        if not math.isclose(per_rank_payload[rank], expected, rel_tol=1e-12, abs_tol=1e-9):
            raise ModelingError(f"ring_allreduce_payload_conservation_failed:{rank}")


def lower_collective(
    call,
    actual_group: Sequence[int],
    topology,
    api_bytes_per_rank: int | float,
    *,
    stack: str = "default",
    expansion_mode: str | None = None,
    band_override: CollectiveBandSpec | None = None,
    peer_matrix: Sequence[Sequence[int]] | None = None,
    send_splits: Sequence[int] | None = None,
    recv_splits: Sequence[int] | None = None,
    communicator_id: str = "",
    coset_id: str = "",
    dtype: str | None = None,
) -> LoweredCollective:
    """Lower a collective call into framed physical phases, hops, and port keys."""
    collective = _normalize_collective_name(call)
    group = _normalize_group(actual_group)
    count = len(group)
    element_bytes = _resolve_element_bytes(call, dtype=dtype)
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

    band = band_override if band_override is not None else select_band(
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
        concurrent = band.algorithm in _CONCURRENT_MESH_ALGORITHMS
        for s_idx, src in enumerate(group):
            self_copies[src] = float(matrix[s_idx][s_idx])
            for step in range(count - 1):
                d_idx = (s_idx + step + 1) % count
                amount = matrix[s_idx][d_idx]
                if amount == 0:
                    continue
                dst = group[d_idx]
                per_rank_payload[src] += float(amount)
                r_idx = 0 if concurrent else step
                transfers.append(_build_transfer("alltoallv", r_idx, src, dst, amount, band, topology))
        rounds = 1 if concurrent else max(count - 1, 1)
        phases = (LoweredPhase("alltoallv", rounds, tuple(transfers), phase_startup),) if transfers else ()
    elif api_bytes == 0.0 or count <= 1:
        phases = ()
    elif collective in {"allreduce", "reduce_scatter", "allgather"}:
        phases = _lower_ring_or_alg(
            collective, group, api_bytes, band, topology, phase_startup, per_rank_payload,
            element_bytes=element_bytes,
        )
        if collective == "allreduce":
            _validate_allreduce_conservation(
                group, api_bytes, band, per_rank_payload, element_bytes,
            )
    else:
        # Balanced AllToAll: total send buffer B splits B/P per peer (including self)
        per_peer = api_bytes / float(count)
        transfers = []
        concurrent = band.algorithm in _CONCURRENT_MESH_ALGORITHMS
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
        namespace=str(getattr(topology, "namespace", "") or ""),
    )


def _effective_hop_flow_cap(transfer: LoweredTransfer, hop: LoweredHop, has_per_hop_caps: bool) -> float | None:
    if hop.flow_cap_bytes_per_ms is not None:
        return float(hop.flow_cap_bytes_per_ms)
    if not has_per_hop_caps and transfer.flow_cap_bytes_per_ms is not None:
        return float(transfer.flow_cap_bytes_per_ms)
    return None


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
) -> tuple[list, list]:
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
    has_per_hop_caps = any(h.flow_cap_bytes_per_ms is not None for h in transfer.hops)
    hop0_svcs = []
    done = []
    for c_idx, chunk_wire in enumerate(chunks):
        alloc = (
            Allocation(f"{prefix}:c{c_idx}:cr", pool_name, min(chunk_wire, credit_cap), (0,))
            if pool_name is not None and chunk_wire > 0.0
            else None
        )
        hop_cursor = gate
        first_hop_svc = gate
        for h_idx, hop in enumerate(transfer.hops):
            demand = {res: chunk_wire for res in hop.resources}
            tx_inj = ub_tx_res.get(hop.source)
            rx_inj = ub_rx_res.get(hop.destination)
            if tx_inj is not None:
                demand[tx_inj] = chunk_wire
            if rx_inj is not None:
                demand[rx_inj] = chunk_wire
            hop_cap = _effective_hop_flow_cap(transfer, hop, has_per_hop_caps)
            if hop_cap is not None:
                sdma_ms = chunk_wire / hop_cap
                sdma_tx = flow_tx_res.get(hop.source)
                sdma_rx = flow_rx_res.get(hop.destination)
                if sdma_tx is not None:
                    demand[sdma_tx] = sdma_ms
                if sdma_rx is not None:
                    demand[sdma_rx] = sdma_ms
            svc = engine.submit(
                f"{prefix}:c{c_idx}:h{h_idx}:svc",
                demand,
                after=(hop_cursor,),
                acquire=(alloc,) if (h_idx == 0 and alloc is not None) else (),
                owner=owner,
                coupled=len(demand) > 1,
            )
            if h_idx == 0:
                first_hop_svc = svc
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
        hop0_svcs.append(first_hop_svc)
        done.append(hop_cursor)
    return hop0_svcs, done


def _prepare_collective_resources(
    engine, name: str, lowered: LoweredCollective, credit_cap: float
) -> tuple[dict[tuple[int, int] | int, str], dict[int, str], dict[int, str], dict[int, str], dict[int, str]]:
    credit_pools: dict[tuple[int, int] | int, str] = {}
    new_pools: dict[str, float] = {}
    flow_tx_res: dict[int, str] = {}
    flow_rx_res: dict[int, str] = {}
    new_resources: list[Resource] = []
    added_resources: set[str] = set()
    ns = str(getattr(lowered, "namespace", "") or "")
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
            has_per_hop_caps = any(h.flow_cap_bytes_per_ms is not None for h in transfer.hops)
            for hop in transfer.hops:
                if _effective_hop_flow_cap(transfer, hop, has_per_hop_caps) is None:
                    continue
                if hop.source not in flow_tx_res:
                    tx_name = sdma_flow_resource_name(hop.source, "tx", namespace=ns)
                    flow_tx_res[hop.source] = tx_name
                    if tx_name not in engine.resources and tx_name not in added_resources:
                        added_resources.add(tx_name)
                        new_resources.append(Resource(tx_name, 1.0, "ms"))
                if hop.destination not in flow_rx_res:
                    rx_name = sdma_flow_resource_name(hop.destination, "rx", namespace=ns)
                    flow_rx_res[hop.destination] = rx_name
                    if rx_name not in engine.resources and rx_name not in added_resources:
                        added_resources.add(rx_name)
                        new_resources.append(Resource(rx_name, 1.0, "ms"))
    ub_tx_res: dict[int, str] = {}
    ub_rx_res: dict[int, str] = {}
    if lowered.die_injection_ceiling_bytes_per_ms is not None:
        ceil = float(lowered.die_injection_ceiling_bytes_per_ms)
        for rank in lowered.group:
            tx_name = die_injection_resource_name(rank, "tx", namespace=ns)
            rx_name = die_injection_resource_name(rank, "rx", namespace=ns)
            ub_tx_res[rank] = tx_name
            ub_rx_res[rank] = rx_name
            if tx_name not in engine.resources and tx_name not in added_resources:
                added_resources.add(tx_name)
                new_resources.append(Resource(tx_name, ceil, "bytes"))
            if rx_name not in engine.resources and rx_name not in added_resources:
                added_resources.add(rx_name)
                new_resources.append(Resource(rx_name, ceil, "bytes"))
    if new_resources or new_pools:
        engine.extend(tuple(new_resources), pools=new_pools or None)
    return credit_pools, ub_tx_res, ub_rx_res, flow_tx_res, flow_rx_res


def _submit_ring_phases(
    engine,
    name: str,
    lowered: LoweredCollective,
    rendezvous,
    *,
    credit_cap: float,
    credit_pools: Mapping[tuple[int, int] | int, str],
    hop_startup_ms: float,
    ub_tx_res: Mapping[int, str],
    ub_rx_res: Mapping[int, str],
    flow_tx_res: Mapping[int, str],
    flow_rx_res: Mapping[int, str],
    owner: str,
) -> dict[int, object]:
    """Schedule ring/double-ring rounds with rank-local send/predecessor causality."""
    is_double_ring = str(lowered.band.algorithm).strip().lower() == "double_ring"
    last_send: dict[tuple[int, int], object] = {}
    last_recv: dict[tuple[int, int], object] = {}
    phase_default = rendezvous
    for p_idx, phase in enumerate(lowered.phases):
        if phase.startup_us > 0.0:
            startup_ms = phase.startup_us / MS_PER_S
            phase_default = engine.submit(
                f"{name}:p{p_idx}:{phase.name}:start",
                after=(phase_default,),
                latency=startup_ms,
                owner=owner,
            )
            for key in sorted(set(last_send) | set(last_recv)):
                deps = tuple(dict.fromkeys(
                    ev for ev in (last_send.get(key), last_recv.get(key)) if ev is not None
                )) or (rendezvous,)
                p_start = engine.submit(
                    f"{name}:p{p_idx}:r{key[0]}:l{key[1]}:start",
                    after=deps,
                    latency=startup_ms,
                    owner=owner,
                )
                last_send[key] = p_start
                last_recv[key] = p_start
        by_round: dict[int, list[LoweredTransfer]] = {}
        for transfer in phase.transfers:
            by_round.setdefault(transfer.round_index, []).append(transfer)
        for round_index in sorted(by_round):
            prev_send, prev_recv = dict(last_send), dict(last_recv)
            next_send: dict[tuple[int, int], list] = {}
            next_recv: dict[tuple[int, int], list] = {}
            for t_idx, transfer in enumerate(by_round[round_index]):
                lane = transfer.lane_index if is_double_ring else 0
                s_key = (transfer.source, lane)
                d_key = (transfer.destination, lane)
                s_dep = prev_send.get(s_key, phase_default)
                r_dep = prev_recv.get(s_key, phase_default)
                t_gate = (
                    s_dep if s_dep is r_dep
                    else engine.submit(
                        f"{name}:p{p_idx}:r{round_index}:t{t_idx}:gate",
                        after=(s_dep, r_dep),
                        owner=owner,
                    )
                )
                pool_key = (
                    (transfer.source, transfer.destination)
                    if lowered.uses_calibrated_startup else transfer.destination
                )
                hop0_svcs, arrivals = _submit_transfer_chunks(
                    engine, f"{name}:p{p_idx}:r{round_index}:t{t_idx}", transfer, t_gate,
                    band=lowered.band, credit_cap=credit_cap,
                    pool_name=credit_pools.get(pool_key),
                    hop_startup_ms=hop_startup_ms,
                    uses_calibrated_startup=lowered.uses_calibrated_startup,
                    ub_tx_res=ub_tx_res, ub_rx_res=ub_rx_res,
                    flow_tx_res=flow_tx_res, flow_rx_res=flow_rx_res, owner=owner,
                )
                next_send.setdefault(s_key, []).extend(hop0_svcs)
                next_recv.setdefault(d_key, []).extend(arrivals)
            for key, evs in next_send.items():
                prior = (prev_send[key],) if key in prev_send else ()
                send_deps = tuple(dict.fromkeys((*prior, *evs)))
                last_send[key] = (
                    send_deps[0] if len(send_deps) == 1
                    else engine.submit(
                        f"{name}:p{p_idx}:r{round_index}:s{key[0]}_l{key[1]}",
                        after=send_deps,
                        owner=owner,
                    )
                )
            for key, evs in next_recv.items():
                prior = (prev_recv[key],) if key in prev_recv else ()
                recv_deps = tuple(dict.fromkeys((*prior, *evs)))
                last_recv[key] = (
                    recv_deps[0] if len(recv_deps) == 1
                    else engine.submit(
                        f"{name}:p{p_idx}:r{round_index}:d{key[0]}_l{key[1]}",
                        after=recv_deps,
                        owner=owner,
                    )
                )
    rank_done: dict[int, object] = {}
    for rank in lowered.group:
        deps = tuple(dict.fromkeys(
            ev for mapping in (last_send, last_recv)
            for (r, _lane), ev in mapping.items()
            if r == rank
        )) or (phase_default,)
        rank_done[rank] = (
            deps[0] if len(deps) == 1
            else engine.submit(f"{name}:r{rank}:complete", after=deps, owner=owner)
        )
    return rank_done


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
            active_slices = [s_idx for s_idx in slice_ids if by_slice.get(s_idx)]
            if not active_slices:
                continue
            first_cursor = slice_cursor[active_slices[0]]
            phase_start = (
                engine.submit(
                    f"{name}:p{p_idx}:{phase.name}:start",
                    after=(first_cursor,),
                    latency=phase.startup_us / MS_PER_S,
                    owner=owner,
                )
                if phase.startup_us > 0.0
                else None
            )
            for s_idx in active_slices:
                s_transfers = by_slice[s_idx]
                upstream = slice_cursor[s_idx]
                if phase_start is None:
                    s_gate = upstream
                elif upstream is first_cursor:
                    s_gate = phase_start
                else:
                    s_gate = engine.submit(
                        f"{name}:p{p_idx}:{phase.name}:s{s_idx}:gate",
                        after=(upstream, phase_start),
                        owner=owner,
                    )
                s_done = []
                for t_idx, transfer in enumerate(s_transfers):
                    pool_key = (
                        (transfer.source, transfer.destination)
                        if lowered.uses_calibrated_startup else transfer.destination
                    )
                    _, arrivals = _submit_transfer_chunks(
                        engine, f"{name}:p{p_idx}:s{s_idx}:t{t_idx}", transfer, s_gate,
                        band=lowered.band, credit_cap=credit_cap,
                        pool_name=credit_pools.get(pool_key),
                        hop_startup_ms=hop_startup_ms,
                        uses_calibrated_startup=lowered.uses_calibrated_startup,
                        ub_tx_res=ub_tx_res, ub_rx_res=ub_rx_res,
                        flow_tx_res=flow_tx_res, flow_rx_res=flow_rx_res, owner=owner,
                    )
                    s_done.extend(arrivals)
                slice_cursor[s_idx] = engine.submit(
                    f"{name}:p{p_idx}:s{s_idx}:done",
                    after=tuple(s_done),
                    owner=owner,
                )
        complete = engine.submit(f"{name}:complete", after=tuple(slice_cursor.values()), owner=owner)
        return {rank: complete for rank in lowered.group}

    if str(lowered.band.algorithm).strip().lower() in {"ring", "double_ring", "pairwise"}:
        return _submit_ring_phases(
            engine, name, lowered, rendezvous,
            credit_cap=credit_cap, credit_pools=credit_pools,
            hop_startup_ms=hop_startup_ms,
            ub_tx_res=ub_tx_res, ub_rx_res=ub_rx_res,
            flow_tx_res=flow_tx_res, flow_rx_res=flow_rx_res,
            owner=owner,
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
                pool_key = (
                    (transfer.source, transfer.destination)
                    if lowered.uses_calibrated_startup else transfer.destination
                )
                _, arrivals = _submit_transfer_chunks(
                    engine, f"{name}:p{p_idx}:r{round_index}:t{t_idx}", transfer, round_gate,
                    band=lowered.band, credit_cap=credit_cap,
                    pool_name=credit_pools.get(pool_key),
                    hop_startup_ms=hop_startup_ms,
                    uses_calibrated_startup=lowered.uses_calibrated_startup,
                    ub_tx_res=ub_tx_res, ub_rx_res=ub_rx_res,
                    flow_tx_res=flow_tx_res, flow_rx_res=flow_rx_res, owner=owner,
                )
                round_done.extend(arrivals)
            round_gate = engine.submit(
                f"{name}:p{p_idx}:r{round_index}:done",
                after=tuple(round_done),
                owner=owner,
            )
        cursor = round_gate

    complete = engine.submit(f"{name}:complete", after=(cursor,), owner=owner)
    return {rank: complete for rank in lowered.group}
