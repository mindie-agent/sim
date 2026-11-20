"""Node-local dense-subgraph TP proofs and union cohorts.

[Role]
  Build and validate the real physical rank groups that carry node-local dense
  projection collectives under attention DP.  A group is confined to one node,
  PP stage and PCP lane; distinct enabled widths may overlap.  The scheduling
  cohort is the connected component of the union of EP and every enabled local
  group, so a local exchange that crosses an EP cohort boundary stays in one
  causal cohort.

[Invariants]
  * The initial supported configuration is global ``tp=1`` and no CP/DCP
    subdivision.  Widths are positive, bounded by node ranks and ``dp*tp``, and
    divide the eligible ranks in every actual node/lane.
  * Every rank belongs to exactly one group per enabled width and no group is
    silently moved across a node boundary.
  * ``R=0``/width-one deployments keep the historical EP partition exactly.

[Timeouts & Locks]
  Bounded rank arithmetic over the declared die table; no I/O or locks.

[Side-Effect Policy]
  Read-only; returns immutable tuples and raises descriptive errors.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
FIRST = 0
SECOND = 1
LOCAL_TP_AXIS = "local_tp"


def _enabled_widths(parallel: Any) -> tuple[int, ...]:
    return tuple(sorted({
        int(parallel.o_proj_tp), int(parallel.linear_proj_tp),
        int(parallel.lm_head_tp),
    } - {ONE}))


def _rank_order(machine: Any) -> tuple[int, ...]:
    return tuple(sorted(int(device.rank) for device in machine.dies))


def _device_map(machine: Any) -> dict[int, Any]:
    return {int(device.rank): device for device in machine.dies}


def _stage_rank_buckets(
    machine: Any, parallel: Any, *, stage: int,
) -> dict[tuple[int, int, int], list[int]]:
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, parallel)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    ext_dp = max(ONE, int(getattr(parallel, "external_dp", ONE) or ONE))
    internal_replicas = max(ONE, int(parallel.dp) * int(parallel.active_pcp_lanes))
    devices = _device_map(machine)
    for ext in range(ext_dp):
        for local_rep in range(internal_replicas):
            replica = ext * internal_replicas + local_rep
            ranks = topology.ranks(replica, stage)
            if not ranks:
                if ext == ZERO:
                    continue
                raise ModelingError(f"local_tp_external_replica_missing:{ext}:stage{stage}")
            for position, rank in enumerate(ranks):
                device = devices.get(int(rank))
                if device is None:
                    raise ModelingError(f"local_tp_rank_undeclared:{rank}")
                lane = (position // max(ONE, int(parallel.tp))
                        if int(parallel.pcp) > ONE else ZERO)
                key = (ext, int(getattr(device, "node", ZERO)), lane)
                buckets.setdefault(key, []).append(int(rank))
    return buckets


def local_tp_groups(
    machine: Any, parallel: Any, *, stage: int, width: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition actual external-DP/node/stage/PCP ranks into consecutive groups of width."""
    if isinstance(width, bool) or not isinstance(width, int) or width < ONE:
        raise ModelingError(f"local_tp_width_invalid:{width}")
    groups: list[tuple[int, ...]] = []
    for (ext, node, lane), ranks in sorted(_stage_rank_buckets(machine, parallel, stage=stage).items()):
        del ext
        ordered = tuple(sorted(ranks))
        if width > len(ordered):
            raise ModelingError(
                f"local_tp_width_exceeds_node_lane:{width}>{len(ordered)}")
        if len(ordered) % width != ZERO:
            raise ModelingError(
                f"local_tp_width_not_divisor:{width}!|{len(ordered)}:node{node}:lane{lane}")
        for start in range(ZERO, len(ordered), width):
            groups.append(ordered[start:start + width])
    if not groups:
        raise ModelingError(f"local_tp_stage_has_no_ranks:{stage}")
    return tuple(groups)


def _union_components(keys: Iterable[int], edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    parent = {int(k): int(k) for k in keys}

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            nxt = parent[node]
            parent[node] = root
            node = nxt
        return root

    for left, right in edges:
        if left in parent and right in parent:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)
    buckets: dict[int, list[int]] = {}
    for node in sorted(parent):
        buckets.setdefault(find(node), []).append(node)
    return tuple(tuple(sorted(items)) for _, items in sorted(buckets.items()))


def local_tp_cohorts(
    machine: Any, parallel: Any,
) -> tuple[tuple[int, ...], ...]:
    """Connected components of the union of EP and enabled local TP groups."""
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, parallel)
    edges: list[tuple[int, int]] = []
    ext_dp = max(ONE, int(getattr(parallel, "external_dp", ONE) or ONE))
    replicas = max(ONE, ext_dp * int(parallel.dp) * int(parallel.active_pcp_lanes))
    for stage in range(max(ONE, int(parallel.pp))):
        for replica in range(replicas):
            for group in topology.groups("ep", replica, stage):
                edges.extend((group[ZERO], rank) for rank in group[ONE:])
        for width in _enabled_widths(parallel):
            for group in local_tp_groups(machine, parallel, stage=stage, width=width):
                edges.extend((group[ZERO], rank) for rank in group[ONE:])
    return _union_components(_rank_order(machine), edges)


def local_tp_replica_cohorts(
    machine: Any, parallel: Any,
) -> tuple[tuple[int, ...], ...]:
    """Projection of physical EP + local-TP connected components onto DP replicas."""
    from mindie_sim.prediction.topology import Topology

    topology = Topology(machine, parallel)
    dp = max(ONE, int(parallel.dp))
    edges: list[tuple[int, int]] = []
    for stage in range(max(ONE, int(parallel.pp))):
        owner = {int(rank): r for r in range(dp)
                 for rank in topology.ranks(r, stage)}
        for r in range(dp):
            for group in topology.groups("ep", r, stage):
                reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                edges.extend((reps[ZERO], rep) for rep in reps[ONE:])
        for width in _enabled_widths(parallel):
            for group in local_tp_groups(machine, parallel, stage=stage, width=width):
                reps = [owner[int(rk)] for rk in group if int(rk) in owner]
                edges.extend((reps[ZERO], rep) for rep in reps[ONE:])
    return _union_components(range(dp), edges)


def validate_local_projection_tp(case: Any) -> None:
    """Validate the declared local widths against physical rank ownership."""
    parallel = case.runtime.parallel
    hardware = case.machine
    widths = _enabled_widths(parallel)
    if not widths:
        return
    if int(parallel.tp) > ONE:
        raise ModelingError("local_projection_tp_requires_attention_dp")
    if int(parallel.cp) > ONE or int(parallel.dcp) > ONE:
        raise ModelingError("local_projection_tp_excludes_context_parallel")
    lane_plane = int(parallel.dp) * int(parallel.tp)
    ranks_per_node = max(ONE, int(getattr(hardware, "ranks_per_node", ONE) or ONE))
    hardware.validate_placement(parallel)
    devices = _device_map(hardware)
    declared = {int(device.rank) for device in hardware.dies}
    from mindie_sim.prediction.topology import Topology

    topology = Topology(hardware, parallel)
    ext_dp = max(ONE, int(getattr(parallel, "external_dp", ONE) or ONE))
    expected = {
        rank
        for replica in range(max(ONE, ext_dp * int(parallel.dp) * int(parallel.active_pcp_lanes)))
        for stage in range(max(ONE, int(parallel.pp)))
        for rank in topology.ranks(replica, stage)
    }
    if not expected <= declared:
        raise ModelingError("local_tp_physical_rank_missing")
    for width in widths:
        if width > ranks_per_node:
            raise ModelingError(
                f"local_tp_width_exceeds_ranks_per_node:{width}>{ranks_per_node}")
        if width > lane_plane:
            raise ModelingError(
                f"local_tp_width_exceeds_lane_plane:{width}>{lane_plane}")
        seen: set[int] = set()
        for stage in range(max(ONE, int(parallel.pp))):
            groups = local_tp_groups(hardware, parallel, stage=stage, width=width)
            stage_ranks = [
                rank for group in groups for rank in group
            ]
            if len(stage_ranks) != len(set(stage_ranks)):
                raise ModelingError("local_tp_rank_in_multiple_groups")
            for group in groups:
                signatures = {devices[rank].resource_signature for rank in group}
                if len(signatures) > ONE:
                    raise ModelingError("local_tp_group_spans_heterogeneous_devices")
                nodes = {int(getattr(devices[rank], "node", ZERO)) for rank in group}
                if len(nodes) > ONE:
                    raise ModelingError("local_tp_group_spans_nodes")
            seen.update(stage_ranks)
        if not expected <= seen:
            raise ModelingError("local_tp_group_coverage_incomplete")


def local_tp_spans_dp(parallel: Any) -> bool:
    """True when an enabled local width can cross a DP replica boundary."""
    return int(parallel.dp) > ONE and max(_enabled_widths(parallel), default=ONE) > ONE


def rank_zero_graph_reuse(parallel: Any, replicas: int, has_device_group: bool) -> bool:
    """Legacy rank-zero graph reuse remains valid only without cross-DP groups."""
    return (int(replicas) > ONE and not has_device_group
            and not parallel.ep_spans_dp and not local_tp_spans_dp(parallel))


def expert_tp_groups(
    topology: Any, *, replica: int = ZERO, stage: int = ZERO,
) -> tuple[tuple[int, ...], ...]:
    """Construct expert-TP rank groups from the resolved ``parallel.etp`` width.

    When an explicit EP ``rank_table`` is declared, ranks inside each ``tp``
    slice are grouped by equal EP position so only ranks owning the same
    experts share an expert-TP group.
    """
    parallel = topology.parallel
    sources = tuple(int(rank) for rank in topology.ranks(replica, stage))
    if not sources:
        raise ModelingError(f"expert_tp_stage_has_no_ranks:{stage}")
    tp = max(ONE, int(parallel.tp))
    etp = max(ONE, int(parallel.etp))
    if tp % etp != ZERO:
        raise ModelingError(f"expert_tp_not_divisor_of_tp:{etp}!|{tp}")
    if etp <= ONE:
        return tuple((rank,) for rank in sources)
    table = getattr(parallel, "rank_table", None)
    if table:
        ep_pos = {
            int(rank): pos
            for ep_group in table
            for pos, rank in enumerate(ep_group)
        }
        groups: list[tuple[int, ...]] = []
        for base in range(ZERO, len(sources), tp):
            tp_slice = sources[base : base + tp]
            if all(rank in ep_pos for rank in tp_slice):
                by_pos: dict[int, list[int]] = {}
                for rank in tp_slice:
                    by_pos.setdefault(ep_pos[rank], []).append(rank)
                groups.extend(tuple(members) for members in by_pos.values() if members)
                continue
            stride = tp // etp
            groups.extend(
                group
                for offset in range(stride)
                if (group := tp_slice[offset::stride])
            )
        return tuple(groups)
    stride = tp // etp
    return tuple(
        group
        for base in range(ZERO, len(sources), tp)
        for offset in range(stride)
        if (group := sources[base + offset : base + tp : stride])
    )


def collective_requires_cohort(op: Any, topology: Any) -> bool:
    """True when the op's actual group spans more than one DP replica."""
    metadata = dict(getattr(op, "metadata", ()) or ())
    parallel = topology.parallel
    stage = int(metadata.get("source_stage", getattr(op, "stage", ZERO)))
    replicas = max(ONE, int(parallel.dp) * int(parallel.active_pcp_lanes))
    owner: dict[int, int] = {}
    for replica in range(replicas):
        for rank in topology.ranks(replica, stage):
            owner[int(rank)] = replica
    axis = str(metadata.get("group_axis", "tp"))
    if axis not in {"cp", "etp", "pcp", "ep", "tp", "dp", LOCAL_TP_AXIS}:
        return False
    declared = metadata.get("group_ranks")
    rep = int(metadata.get("replica", ZERO))
    if declared:
        groups: Sequence[Sequence[int]] = (tuple(declared),)
    elif axis == "etp":
        groups = expert_tp_groups(topology, replica=rep, stage=stage)
    elif axis == "dp":
        dp_count = max(ONE, int(parallel.dp))
        rows = [tuple(int(rk) for rk in topology.ranks(r, stage)) for r in range(dp_count)]
        width = min((len(row) for row in rows), default=ZERO)
        groups = tuple(tuple(rows[r][off] for r in range(dp_count)) for off in range(width))
    else:
        groups = tuple(topology.groups(axis, rep, stage, metadata=metadata))
    for group in groups:
        owners = {owner.get(int(rank)) for rank in group if int(rank) in owner}
        owners.discard(None)
        if len(owners) > ONE:
            return True
    return False
