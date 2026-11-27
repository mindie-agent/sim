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
    ext_base = (int(replica) // parallel.dp) * parallel.dp
    placement = None
    if machine is None:
        def rank_at(d, c, t):
            return ((d * parallel.pp + stage) * parallel.pcp + c) * parallel.tp + t
    elif hasattr(machine, "validate_placement") and hasattr(parallel, "world_size"):
        placement = machine.validate_placement(parallel)
        stage_ranks = {
            d: placement.stage_ranks(d, stage)
            for d in range(ext_base, ext_base + parallel.dp)
        }

        def rank_at(d, c, t):
            return stage_ranks[d][c * parallel.tp + t]
    else:
        def rank_at(d, c, t):
            idx = ((d * parallel.pp + stage) * parallel.pcp + c) * parallel.tp + t
            return machine.dies[idx].rank
    coords = {
        rank_at(d, c, t): (d, c, t)
        for d in range(ext_base, ext_base + parallel.dp)
        for c in range(parallel.pcp)
        for t in range(parallel.tp)
    }
    if parallel.rank_table is not None:
        groups = tuple(
            tuple(group) for group in parallel.rank_table
            if all(r in coords for r in group)
        )
    elif placement is not None:
        groups = tuple(dict.fromkeys(
            placement.ep_group_ranks(replica, stage, lane=lane)
            for lane in range(parallel.pcp * parallel.tp)
        ))
    elif machine is not None:
        groups = Topology(machine, parallel).groups("ep", replica, stage)
    else:
        etp = max(1, int(getattr(parallel, "etp", 1) or 1))
        ep_tp_w = max(1, parallel.tp // etp)
        ep_spans_pcp = parallel.pcp > 1 and parallel.ep > parallel.dp * ep_tp_w
        if ep_spans_pcp:
            groups = tuple(
                tuple(
                    rank_at(d, c, t)
                    for d in range(ext_base, ext_base + parallel.dp)
                    for c in range(parallel.pcp)
                    for t in range(off, off + ep_tp_w)
                )[i : i + parallel.ep]
                for off in range(0, parallel.tp, ep_tp_w)
                for i in range(0, parallel.dp * parallel.pcp * ep_tp_w, parallel.ep)
            )
        else:
            groups = tuple(
                tuple(
                    rank_at(d, c, t)
                    for d in range(ext_base, ext_base + parallel.dp)
                    for t in range(off, off + ep_tp_w)
                )[i : i + parallel.ep]
                for c in range(parallel.pcp)
                for off in range(0, parallel.tp, ep_tp_w)
                for i in range(0, parallel.dp * ep_tp_w, parallel.ep)
            )
    return coords, groups


def ep_replica_group(parallel, machine, participants, replica) -> tuple[int, ...]:
    """Return the subset of ``participants`` sharing an EP collective with ``replica``."""
    members = tuple(participants)
    if len(members) <= 1:
        return members
    if parallel.rank_table is not None:
        linked = {int(replica)}
        for stage in range(parallel.pp):
            coords, tables = ep_groups(parallel, machine, replica, stage)
            changed = True
            while changed:
                changed = False
                for table in tables:
                    owners = {coords[r][0] for r in table if r in coords}
                    if owners & linked and not owners <= linked:
                        linked |= owners
                        changed = True
        return tuple(r for r in members if r in linked)
    ep_w = max(1, int(getattr(parallel, "ep_dp_width", 1) or 1))
    first = (int(replica) // ep_w) * ep_w
    return tuple(r for r in members if first <= r < first + ep_w)


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
    topo = Topology(machine, parallel)
    for stage in range(parallel.pp):
        members = topo.ranks(replica, stage)
        if rank in members:
            return members.index(rank) // parallel.tp
    return None


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


def _resolve_stage_coords(parallel, machine, replica, rank):
    """Locate the pipeline stage carrying ``rank`` and return ``(coords, groups, physical)``."""
    coords, groups = ep_groups(parallel, machine, replica, 0)
    if rank is not None and rank not in coords and parallel.pp > 1:
        for stage in range(1, parallel.pp):
            cand_coords, cand_groups = ep_groups(parallel, machine, replica, stage)
            if rank in cand_coords:
                coords, groups = cand_coords, cand_groups
                break
    physical = next(
        (r for r, (d, c, t) in coords.items() if d == replica and c == 0 and t == 0),
        None,
    )
    if rank is not None and rank in coords:
        physical = rank
    return coords, groups, physical


def rank_ledger_for(
    frontend, model, runtime, views, tokens, peers, replica, machine, rank,
    routing_views=None,
) -> RankLedger | None:
    """Compute the single-step rank row ledger for one EP group."""
    del model
    p = runtime.parallel
    coords, groups, physical = _resolve_stage_coords(p, machine, replica, rank)
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
        view_prefill_ranges = tuple(v.query_ranges for v in views if v.phase == "prefill")
        lane = next(
            (
                c for c in lanes
                if tuple(v.query_ranges for v in lane_views(own, p.pcp, c, decode=False))
                == view_prefill_ranges
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
    etp = max(1, int(getattr(p, "etp", 1) or 1))
    ep_tp_w = max(1, p.tp // etp)
    sharded_tp = bool(unit > 1 or runtime.kernels.get("moe_token_partition"))
    scoped_tp = p.tp if sharded_tp else min(ep_tp_w, p.ep)
    for (d, c), members in by_lane.items():
        use_decode = c == 0 or p.pcp_lane_policy in {"replica_lanes", "shard_decode_requests"}
        has_prefill = any(v.phase == "prefill" for v in expanded[d])
        part = lane_views(expanded[d], p.pcp, c, decode=use_decode)
        count_rows = sum(v.q_len for v in part)
        active_rows = sum(v.logical_q_len for v in part)
        parent = count_rows
        if expanded[d] and (use_decode or has_prefill):
            bucket, _ = frontend.graph.select(expanded[d], runtime)
            extra = max(0, int(bucket or 0) - sum(v.q_len for v in expanded[d]))
            if p.pcp > 1 and has_prefill:
                extra = sum(hi - lo for lo, hi in pcp_rank_chunk_ranges(extra, p.pcp, c))
            parent += extra
        represented = (d, c) == coords[physical][:2]
        if represented and tokens is not None:
            if use_decode:
                parent = max(count_rows, tokens)
            elif has_prefill:
                unowned_decode = sum(
                    v.q_len + (0 if v.global_q_len else v.draft_proposal_tokens)
                    for v in views if v.phase != "prefill"
                )
                parent = max(parent, tokens - unowned_decode)
            else:
                parent = count_rows
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
        ordered_members = sorted(members)
        for member_idx, (t, position) in enumerate(ordered_members):
            lane_index = (
                member_idx
                if (not sharded_tp and len(ordered_members) == scoped_tp)
                else (t % scoped_tp)
            )
            if sharded_tp:
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
    "ep_replica_group",
    "lane_views",
    "rank_ledger_for",
    "replica_cohorts",
    "routing_union",
    "selected_lane",
]
