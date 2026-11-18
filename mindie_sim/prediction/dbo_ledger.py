"""Request and rank ownership of overlapping MoE microbatches.

[Role] Project each logical cohort onto the members of one EP collective.
[Invariants] A logical row has one source per EP group. Child capacities add to the parent.
[Timeouts & Locks] Finite rank and request scans, no locks.
[Side-Effect Policy] Reads request views, graph policy and placement.
"""
from collections import defaultdict
from dataclasses import dataclass, replace

from mindie_sim.catalog.context_parallel import pcp_rank_chunk_ranges
from .overlap_plan import MicrobatchPolicy, build_microbatch_regions
from .topology import Topology


@dataclass(frozen=True)
class RegionLedger:
    region: object
    capacity: int
    physical: int
    valid: tuple[int, ...]
    allocated: tuple[int, ...]
    owner: int
    routing: tuple


def ep_groups(parallel, machine, replica, stage):
    if machine is None:
        rank = lambda d, c, t: ((d * parallel.pp + stage) * parallel.pcp + c) * parallel.tp + t
    else:
        rank = lambda d, c, t: machine.dies[
            ((d * parallel.pp + stage) * parallel.pcp + c) * parallel.tp + t].rank
    coords = {rank(d, c, t): (d, c, t) for d in range(parallel.dp)
              for c in range(parallel.pcp) for t in range(parallel.tp)}
    if parallel.rank_table is not None:
        groups = tuple(tuple(group) for group in parallel.rank_table
                       if all(r in coords for r in group))
    elif machine is not None:
        groups = Topology(machine, parallel).groups("ep", replica, stage)
    else:
        groups = tuple(tuple(rank(d, c, t) for d, t in plane[i:i + parallel.ep])
                       for c in range(parallel.pcp)
                       for plane in [[(d, t) for d in range(parallel.dp)
                                      for t in range(parallel.tp)]]
                       for i in range(0, len(plane), parallel.ep))
    return coords, groups


def replica_cohorts(machine, parallel, groups):
    """Connected replica sets must enter every overlapping collective together."""
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
    result = []
    for v in views:
        if v.phase != "prefill" or degree == 1:
            if decode or v.phase == "prefill":
                result.append(v)
            continue
        spans = pcp_rank_chunk_ranges(v.q_len, degree, lane)
        result.append(replace(v, global_q_len=v.q_len, query_ranges=spans,
                              q_len=sum(hi - lo for lo, hi in spans)))
    return tuple(result)


def _capacities(counts, parent, unit):
    slots = [n + (-n) % unit for n in counts]
    budget = parent + (-parent) % unit
    if sum(slots) > budget:
        return ()
    slots[-1] += budget - sum(slots)
    return slots


def plan(frontend, model, runtime, views, tokens, peers, replica, machine, rank,
         count, routing_views=None):
    p = runtime.parallel
    coords, groups = ep_groups(p, machine, replica, 0)
    physical = next((r for r, (d, c, t) in coords.items()
                     if d == replica and c == 0 and t == 0), None)
    if rank is not None and rank in coords:
        physical = rank
    if physical is None:
        return ()
    if peers is None:
        local = tuple(replace(v, q_len=v.global_q_len or v.q_len,
                              draft_proposal_tokens=0) for v in views)
        peers = {replica: local}
    else:
        peers = dict(peers) if hasattr(peers, "items") else dict(enumerate(peers))
    expanded = {d: tuple(replace(v, q_len=v.global_q_len or (v.q_len + v.draft_proposal_tokens),
                                 query_ranges=(), global_q_len=0)
                         for v in raw) for d, raw in peers.items()}
    own = expanded[replica]
    # The frontend supplied a concrete PCP interval, possibly its critical lane.
    if any(v.phase == "prefill" and v.global_q_len for v in views):
        lanes = (coords[physical][1], *range(p.pcp))
        lane = next((c for c in lanes if tuple(v.query_ranges for v in
                    lane_views(own, p.pcp, c)) == tuple(v.query_ranges for v in views)),
                    coords[physical][1])
        physical = next(r for r, (d, c, t) in coords.items()
                        if (d, c, t) == (replica, lane, coords[physical][2]))
    group = next((g for g in groups if physical in g), ())
    if not group or any(coords[r][0] not in expanded for r in group):
        return ()
    by_lane = defaultdict(list)
    for position, r in enumerate(group):
        d, c, t = coords[r]
        by_lane[d, c].append((t, position))
    if count > 1 and any(len(expanded[d]) < count for d, _ in by_lane):
        return ()
    policy = MicrobatchPolicy(count=count)
    regions = build_microbatch_regions(views, policy=policy)
    declarations = runtime.routing.get("realizations", ())
    if (count > 1 and any(not (entry.get("requests") or entry.get("request_views"))
                          for entry in declarations)):
        # An unordered histogram does not determine either child's expert rows.
        return ()
    if len(regions) != count:
        return ()
    valid = [[0] * len(group) for _ in regions]
    allocated = [[0] * len(group) for _ in regions]
    capacity = [0] * count
    routing = [[] for _ in regions]
    unit = p.tp if runtime.kernels.get("sequence_parallel") else 1
    for (d, c), members in by_lane.items():
        first_lane = min(other for owner, other in by_lane if owner == d)
        use_decode = c == first_lane or p.pcp_lane_policy == "replica_lanes"
        chunks = [lane_views(expanded[d][i::count], p.pcp, c, decode=use_decode)
                  for i in range(count)]
        counts = [sum(v.q_len for v in part) for part in chunks]
        active = [sum(v.logical_q_len for v in part) for part in chunks]
        parent = sum(counts)
        if expanded[d]:
            bucket, _ = frontend.graph.select(expanded[d], runtime)
            extra = max(0, int(bucket or 0) - sum(v.q_len for v in expanded[d]))
            if p.pcp > 1 and any(v.phase == "prefill" for v in expanded[d]):
                extra = sum(hi - lo for lo, hi in pcp_rank_chunk_ranges(extra, p.pcp, c))
            parent += extra
        represented = (d, c) == coords[physical][:2]
        if represented and tokens is not None:
            parent = max(sum(counts), tokens)
        slots = _capacities(counts, parent, unit)
        if not slots or (unit > 1 and runtime.kernels["sequence_parallel"].get(
                "padding", "none") == "none" and slots != counts):
            return ()
        for i, part in enumerate(chunks):
            if represented:
                capacity[i] = slots[i]
            routing[i].extend(v for v in part if not v.padding)
            scoped_tp = p.tp if unit > 1 or runtime.kernels.get("moe_token_partition") else min(p.tp, p.ep)
            for t, position in members:
                lane_index = t % scoped_tp
                if unit > 1 or runtime.kernels.get("moe_token_partition"):
                    width = (slots[i] + scoped_tp - 1) // scoped_tp
                    owned = max(0, min(width, active[i] - lane_index * width))
                else:
                    q, rem = divmod(slots[i], scoped_tp)
                    width = q + (lane_index < rem)
                    q, rem = divmod(active[i], scoped_tp)
                    owned = q + (lane_index < rem)
                allocated[i][position] = width
                valid[i][position] = owned
    routing = [routing_union(part) for part in routing]
    if routing_views is not None:
        from collections.abc import Mapping
        declared = {v["request_id"] if isinstance(v, Mapping) else v.request_id: v
                    for v in routing_views}
        def correlation(v):
            record = declared.get(v.request_id)
            if record is None:
                return v
            if isinstance(record, Mapping):
                return replace(v, context=record.get("context", v.context))
            return replace(record, q_len=v.q_len, global_q_len=v.global_q_len,
                           query_ranges=v.query_ranges)
        routing = [tuple(correlation(v) for v in part) for part in routing]
    return tuple(RegionLedger(r, capacity[i], sum(allocated[i]), tuple(valid[i]),
                              tuple(allocated[i]), group.index(physical), tuple(routing[i])) for i, r in enumerate(regions))


def selected_lane(parallel, machine, replica, rank):
    if parallel.pcp == 1 or rank is None or machine is None:
        return None
    members = Topology(machine, parallel).ranks(replica, 0)
    return members.index(rank) // parallel.tp if rank in members else None


def routing_union(views):
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
            base = replace(base, query_ranges=tuple(merged),
                           q_len=sum(hi - lo for lo, hi in merged))
        result.append(base)
    return tuple(result)


def partition_runtime(runtime, scope):
    if not runtime.routing.get("realizations"):
        return runtime
    members = {v.request_id: v for v in scope.routing}
    records = []
    for declaration in runtime.routing["realizations"]:
        field = "requests" if "requests" in declaration else "request_views"
        if field not in declaration:
            records.append(declaration)
            continue
        requests = []
        for record in declaration[field]:
            v = members.get(record["request_id"])
            if v is None:
                continue
            item = dict(record)
            if v.query_ranges and item["q_len"] != v.q_len:
                ids = item["topk_ids"]
                original = item.get("query_ranges") or ((0, item["q_len"]),)
                previous = {position: i for i, position in enumerate(
                    p for lo, hi in original for p in range(lo, hi))}
                indices = tuple(previous[p] for lo, hi in v.query_ranges for p in range(lo, hi))
                item.update(q_len=v.q_len, query_ranges=v.query_ranges,
                            topk_ids=tuple(ids[i] for i in indices))
                if "source_owners" in item:
                    item["source_owners"] = tuple(item["source_owners"][i] for i in indices)
            requests.append(item)
        if requests:
            records.append({**declaration, field: requests,
                            "tokens": sum(v["q_len"] for v in requests)})
    return replace(runtime, routing={"realizations": records})
