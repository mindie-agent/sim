"""Rank and cohort ownership ledger for EP, PCP, and sequence-parallel groups.

[Role] Project one logical step's request views onto physical EP/PCP ranks and
  cohorts without graph-level microbatch splitting.
[Invariants] Every logical row has a single owning source rank per EP group.
  Collective participant cohorts stay closed under declared rank tables.
[Timeouts & Locks] Finite rank and request scans; no I/O or locks.
[Side-Effect Policy] Reads request views, parallel topology and placement;
  returns immutable ``RankLedger`` and tuple records.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

from mindie_sim.catalog.context_parallel import pcp_rank_chunk_ranges
from mindie_sim.prediction.topology import Topology


@dataclass(frozen=True)
class RankLedger:
    """Single-step physical row ownership across one EP collective group."""

    capacity: int
    physical: int
    valid: tuple[int, ...]
    allocated: tuple[int, ...]
    owner: int
    routing: tuple


def ep_groups(parallel, machine, replica, stage):
    """Return ``(rank_to_coords, ep_group_tuples)`` for one pipeline stage."""
    if machine is None:
        def rank_at(d, c, t):
            return ((d * parallel.pp + stage) * parallel.pcp + c) * parallel.tp + t
    else:
        def rank_at(d, c, t):
            idx = ((d * parallel.pp + stage) * parallel.pcp + c) * parallel.tp + t
            return machine.dies[idx].rank
    coords = {
        rank_at(d, c, t): (d, c, t)
        for d in range(parallel.dp)
        for c in range(parallel.pcp)
        for t in range(parallel.tp)
    }
    if parallel.rank_table is not None:
        groups = tuple(
            tuple(group) for group in parallel.rank_table
            if all(r in coords for r in group)
        )
    elif machine is not None:
        groups = Topology(machine, parallel).groups("ep", replica, stage)
    else:
        plane = [(d, t) for d in range(parallel.dp) for t in range(parallel.tp)]
        groups = tuple(
            tuple(rank_at(d, c, t) for d, t in plane[i : i + parallel.ep])
            for c in range(parallel.pcp)
            for i in range(0, len(plane), parallel.ep)
        )
    return coords, groups


def replica_cohorts(machine, parallel, groups):
    """Merge replica groups that share any rank-table EP collective."""
    if parallel.rank_table is None:
        return groups
    sets = [set(group) for group in groups]
    for stage in range(parallel.pp):
        coords, tables = ep_groups(parallel, machine, 0, stage)
        for table in tables:
            owners = {coords[r][0] for r in table}
            for group in sets:
                if group & owners:
                    owners |= group
            sets = [group for group in sets if not group & owners] + [owners]
    return sorted(tuple(sorted(group)) for group in sets)


def lane_views(views, degree, lane, *, decode=True):
    """Project prefill request views onto one PCP context-parallel lane."""
    result = []
    for view in views:
        if view.phase != "prefill" or degree == 1:
            if decode or view.phase == "prefill":
                result.append(view)
            continue
        spans = pcp_rank_chunk_ranges(view.q_len, degree, lane)
        result.append(replace(
            view,
            global_q_len=view.q_len,
            query_ranges=spans,
            q_len=sum(hi - lo for lo, hi in spans),
        ))
    return tuple(result)


def selected_lane(parallel, machine, replica, rank):
    """Return the PCP lane index of ``rank`` within ``replica``, or ``None``."""
    if parallel.pcp == 1 or rank is None or machine is None:
        return None
    members = Topology(machine, parallel).ranks(replica, 0)
    return members.index(rank) // parallel.tp if rank in members else None


def routing_union(views):
    """Merge PCP query ranges belonging to the same request into one view."""
    groups = defaultdict(list)
    for view in views:
        groups[view.request_id].append(view)
    result = []
    for members in groups.values():
        base = members[0]
        if base.global_q_len and base.phase == "prefill":
            merged = []
            for lo, hi in sorted({r for view in members for r in view.query_ranges}):
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
                elif hi > lo:
                    merged.append((lo, hi))
            base = replace(
                base,
                query_ranges=tuple(merged),
                q_len=sum(hi - lo for lo, hi in merged),
            )
        result.append(base)
    return tuple(result)


def rank_ledger_for(
    frontend, model, runtime, views, tokens, peers, replica, machine, rank,
    routing_views=None,
) -> RankLedger | None:
    """Compute the single-step rank row ledger for one EP group."""
    del model
    p = runtime.parallel
    coords, groups = ep_groups(p, machine, replica, 0)
    physical = next(
        (r for r, (d, c, t) in coords.items() if d == replica and c == 0 and t == 0),
        None,
    )
    if rank is not None and rank in coords:
        physical = rank
    if physical is None:
        return None
    if peers is None:
        local = tuple(
            replace(v, q_len=v.global_q_len or v.q_len, draft_proposal_tokens=0)
            for v in views
        )
        peers = {replica: local}
    else:
        peers = dict(peers) if hasattr(peers, "items") else dict(enumerate(peers))
    expanded = {
        d: tuple(
            replace(
                v,
                q_len=v.global_q_len or (v.q_len + v.draft_proposal_tokens),
                query_ranges=(),
                global_q_len=0,
            )
            for v in raw
        )
        for d, raw in peers.items()
    }
    own = expanded[replica]
    if any(v.phase == "prefill" and v.global_q_len for v in views):
        lanes = (coords[physical][1], *range(p.pcp))
        lane = next(
            (
                c for c in lanes
                if tuple(v.query_ranges for v in lane_views(own, p.pcp, c))
                == tuple(v.query_ranges for v in views)
            ),
            coords[physical][1],
        )
        physical = next(
            r for r, (d, c, t) in coords.items()
            if (d, c, t) == (replica, lane, coords[physical][2])
        )
    group = next((g for g in groups if physical in g), ())
    if not group or any(coords[r][0] not in expanded for r in group):
        return None
    by_lane = defaultdict(list)
    for position, r in enumerate(group):
        d, c, t = coords[r]
        by_lane[d, c].append((t, position))
    if not views:
        return None
    valid = [0] * len(group)
    allocated = [0] * len(group)
    capacity = 0
    routing = []
    unit = p.tp if runtime.kernels.get("sequence_parallel") else 1
    for (d, c), members in by_lane.items():
        first_lane = min(other for owner, other in by_lane if owner == d)
        use_decode = c == first_lane or p.pcp_lane_policy == "replica_lanes"
        part = lane_views(expanded[d], p.pcp, c, decode=use_decode)
        count_rows = sum(v.q_len for v in part)
        active_rows = sum(v.logical_q_len for v in part)
        parent = count_rows
        if expanded[d]:
            bucket, _ = frontend.graph.select(expanded[d], runtime)
            extra = max(0, int(bucket or 0) - sum(v.q_len for v in expanded[d]))
            if p.pcp > 1 and any(v.phase == "prefill" for v in expanded[d]):
                extra = sum(hi - lo for lo, hi in pcp_rank_chunk_ranges(extra, p.pcp, c))
            parent += extra
        represented = (d, c) == coords[physical][:2]
        if represented and tokens is not None:
            parent = max(count_rows, tokens)
        slot = parent + (-parent) % unit
        if (
            unit > 1
            and runtime.kernels["sequence_parallel"].get("padding", "none") == "none"
            and slot != count_rows
        ):
            return None
        if represented:
            capacity = slot
        routing.extend(v for v in part if not v.padding)
        scoped_tp = (
            p.tp if unit > 1 or runtime.kernels.get("moe_token_partition")
            else min(p.tp, p.ep)
        )
        for t, position in members:
            lane_index = t % scoped_tp
            if unit > 1 or runtime.kernels.get("moe_token_partition"):
                width = (slot + scoped_tp - 1) // scoped_tp
                owned = max(0, min(width, active_rows - lane_index * width))
            else:
                q, rem = divmod(slot, scoped_tp)
                width = q + (lane_index < rem)
                q, rem = divmod(active_rows, scoped_tp)
                owned = q + (lane_index < rem)
            allocated[position] = width
            valid[position] = owned
    merged_routing = routing_union(routing)
    if routing_views is not None:
        declared = {
            v["request_id"] if isinstance(v, Mapping) else v.request_id: v
            for v in routing_views
        }

        def correlate(v):
            record = declared.get(v.request_id)
            if record is None:
                return v
            if isinstance(record, Mapping):
                return replace(v, context=record.get("context", v.context))
            return replace(
                record,
                q_len=v.q_len,
                global_q_len=v.global_q_len,
                query_ranges=v.query_ranges,
            )

        merged_routing = tuple(correlate(v) for v in merged_routing)
    return RankLedger(
        capacity=capacity,
        physical=sum(allocated),
        valid=tuple(valid),
        allocated=tuple(allocated),
        owner=group.index(physical),
        routing=merged_routing,
    )


__all__ = [
    "RankLedger",
    "ep_groups",
    "lane_views",
    "rank_ledger_for",
    "replica_cohorts",
    "routing_union",
    "selected_lane",
]
