"""Plan and map common SplitCore ownership for q1 dense MLA.

[Role] Reproduce the restricted source host planner and translate its endpoints.
[Invariants] AIC endpoints are exclusive; FD AIV endpoints are inclusive.
  Every KV tile and every split output head belongs to exactly one owner.
  Host metadata is bound to context, physical heads and physical core count.
[Timeouts & Locks] O(batch + owners + FD heads), no tile expansion or locks.
[Side-Effect Policy] Pure metadata; never compiles source or reads observations.
"""
from collections import Counter
from dataclasses import dataclass
from math import isqrt

from mindie_sim.schema.common import ModelingError


KEY_TILE = 512
_FD_HEAD_TILE = 8
_M_TILE_COST = 6
_S2_BLOCK_COST = 10
_S2_COST_BLOCK = 64


def source_dispatch(contexts, heads, cores):
    """Dense q1 specialization of common SplitCore, without a timing objective.

    A single N2 and M<256 give one M-tail row per sequence. AssignByBatch
    and AssignByRow consequently have identical remaining work: once the
    whole remaining sequence does not fit, only AssignByBlock/ForceAssign
    can advance it. Integer host costs are dimension scores, never time.
    """
    sizes = tuple((c+KEY_TILE)//KEY_TILE for c in contexts)
    tail_keys = ((c + 1) % KEY_TILE or KEY_TILE for c in contexts)
    tail_costs = tuple(_M_TILE_COST + _S2_BLOCK_COST * ((keys + _S2_COST_BLOCK - 1) // _S2_COST_BLOCK)
                       for keys in tail_keys)
    normal_cost = _M_TILE_COST + _S2_BLOCK_COST * (KEY_TILE // _S2_COST_BLOCK)
    costs = tuple((size-1)*normal_cost+tail for size,tail in zip(sizes,tail_costs))
    total = sum(costs)
    maximum = min(cores,sum(sizes))
    # For every count below the physical-core cap this integer form equals
    # uint32(sqrt(float(totalBlocks)+.25f)+.5f), without float range loss.
    minimum = min((isqrt(2**2*sum(sizes)+1)+1)//2,maximum)
    best_ends, best_cost = None, None
    for candidate in range(minimum,maximum+1):
        seq = tile = 0
        remaining, maximum_cost, ends = total, 0, []
        for owner in range(candidate):
            if seq == len(sizes):
                break
            limit = remaining//(candidate-owner)
            assigned = count = 0
            # Source half-tail tolerance applies when assigning a whole
            # remaining sequence, including one begun by the previous owner.
            while seq < len(sizes):
                whole = costs[seq]-tile*normal_cost
                if assigned+whole > limit+tail_costs[seq]//2:
                    break
                assigned += whole
                count += sizes[seq]-tile
                seq, tile = seq+1, 0
            if seq < len(sizes):
                current = tail_costs[seq] if tile == sizes[seq]-1 else normal_cost
                # AssignByBlock fixes curCost for this run. A cheaper final
                # tile could only be consumed by the preceding whole-row step.
                take = max(0,(limit+current//2-assigned)//current)
                assigned += take*current
                count += take
                tile += take
                if count == 0:
                    assigned, tile = current, tile+1
                    if tile == sizes[seq]:
                        seq, tile = seq+1, 0
            remaining -= assigned
            maximum_cost = max(maximum_cost,assigned)
            ends.append([seq,0,tile])
        if best_cost is None or maximum_cost < best_cost:
            best_cost, best_ends = maximum_cost, ends

    counts, previous = Counter(), (0,0)
    for seq,_,tile in best_ends:
        for index in range(previous[0],min(seq+1,len(sizes))):
            start = previous[1] if index == previous[0] else 0
            end = tile if index == seq else sizes[index]
            if end > start:
                counts[index] += 1
        previous = seq,tile
    parts = (heads+_FD_HEAD_TILE - 1)//_FD_HEAD_TILE
    fd = [[seq,0,splits,parts,heads-(parts-1)*_FD_HEAD_TILE]
          for seq,splits in sorted(counts.items()) if splits > 1]
    fd_ends = []
    if fd:
        load = sum(row[2]*row[3] for row in fd)
        vectors = min(sum(row[3] for row in fd),2*len(best_ends))
        threshold, current, previous = load/vectors, 0, (0,0)
        for task,(_,_,splits,parts,_) in enumerate(fd):
            for part in range(parts):
                if splits > (threshold-current)*2:
                    fd_ends.append(list(previous))
                    load -= current
                    threshold, current = load/(vectors-len(fd_ends)), 0
                current += splits
                previous = task,part
        fd_ends.append(list(previous))
    return {'cores':cores,'heads':heads,'key_tile':KEY_TILE,'contexts':list(contexts),
            'ends':best_ends,'fd':fd,'fd_ends':fd_ends}


@dataclass(frozen=True)
class Dispatch:
    owners: tuple
    merges: tuple

    @classmethod
    def parse(cls, raw, contexts, heads, cores):
        if raw is None:
            raise ModelingError('paged_mla_arch22_requires_source_host_dispatch')
        if (tuple(raw['contexts']) != contexts or raw['heads'] != heads
                or raw['cores'] != cores or raw['key_tile'] != KEY_TILE):
            raise ModelingError('paged_mla_arch22_dispatch_geometry_mismatch')
        sizes = tuple((c+KEY_TILE)//KEY_TILE for c in contexts)
        owners, previous = [], (0, 0)
        for seq, group, tile in raw['ends']:
            if (group != 0 or not 0 <= seq <= len(sizes) or tile < 0
                    or (seq == len(sizes) and tile != 0)
                    or (seq < len(sizes) and tile >= sizes[seq])
                    or (seq, tile) <= previous):
                raise ModelingError('paged_mla_arch22_invalid_aic_endpoint')
            rows = []
            start_seq, start_tile = previous
            for index in range(start_seq, seq+1):
                if index == len(sizes):
                    break
                start = start_tile if index == start_seq else 0
                end = tile if index == seq else sizes[index]
                if end > start:
                    rows.append((index, start, end-start))
            owners.append(tuple(rows))
            previous = seq, tile
        if (previous != (len(sizes), 0) or not owners or len(owners) > cores
                or any(not rows for rows in owners)):
            raise ModelingError('paged_mla_arch22_aic_coverage')
        counts = Counter(row[0] for rows in owners for row in rows)
        fd = tuple(tuple(row) for row in raw['fd'])
        if ({row[0]: row[2] for row in fd} != {s:n for s,n in counts.items() if n > 1}
                or len({row[0] for row in fd}) != len(fd)
                or any(group != 0 or parts != (heads+_FD_HEAD_TILE - 1)//_FD_HEAD_TILE
                       or tail != heads-(parts-1)*_FD_HEAD_TILE for _,group,_,parts,tail in fd)):
            raise ModelingError('paged_mla_arch22_fd_split_identity')
        merges, previous = [], (0, 0)
        for task, part in raw['fd_ends']:
            if (not 0 <= task < len(fd) or not 0 <= part < fd[task][3]
                    or (task,part) < previous):
                raise ModelingError('paged_mla_arch22_invalid_fd_endpoint')
            rows = []
            for index in range(previous[0], task+1):
                start = previous[1] if index == previous[0] else 0
                end = part+1 if index == task else fd[index][3]
                seq, _, splits, parts, tail = fd[index]
                if end > start:
                    last_head = heads if end == parts else end*_FD_HEAD_TILE
                    rows.append((seq, start*_FD_HEAD_TILE, last_head-start*_FD_HEAD_TILE, splits))
            merges.append(tuple(rows))
            previous = ((task+1,0) if part+1 == fd[task][3] else (task,part+1))
        if (previous != (len(fd),0) or len(merges) > 2*len(owners)
                or any(not rows for rows in merges)):
            raise ModelingError('paged_mla_arch22_fd_head_coverage')
        return cls(tuple(tuple((*row, counts[row[0]]) for row in rows) for rows in owners),
                   tuple(merges))
