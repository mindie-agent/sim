"""Map retained common SplitCore output to q1 dense MLA physical owners.

[Role] Translate source host endpoints, without selecting a split count.
[Invariants] AIC endpoints are exclusive; FD AIV endpoints are inclusive.
  Every KV tile and every split output head belongs to exactly one owner.
  Host metadata is bound to context, physical heads and physical core count.
[Timeouts & Locks] O(batch + owners + FD heads), no tile expansion or locks.
[Side-Effect Policy] Pure metadata; never compiles source or reads observations.
"""
from collections import Counter
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class Dispatch:
    owners: tuple
    merges: tuple

    @classmethod
    def parse(cls, raw, contexts, heads, cores):
        if raw is None:
            raise ModelingError('paged_mla_arch22_requires_source_host_dispatch')
        if (tuple(raw['contexts']) != contexts or raw['heads'] != heads
                or raw['cores'] != cores or raw['key_tile'] != 512):
            raise ModelingError('paged_mla_arch22_dispatch_geometry_mismatch')
        sizes = tuple((c+512)//512 for c in contexts)
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
                or any(group != 0 or parts != (heads+7)//8
                       or tail != heads-(parts-1)*8 for _,group,_,parts,tail in fd)):
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
                    last_head = heads if end == parts else end*8
                    rows.append((seq, start*8, last_head-start*8, splits))
            merges.append(tuple(rows))
            previous = ((task+1,0) if part+1 == fd[task][3] else (task,part+1))
        if (previous != (len(fd),0) or len(merges) > 2*len(owners)
                or any(not rows for rows in merges)):
            raise ModelingError('paged_mla_arch22_fd_head_coverage')
        return cls(tuple(tuple((*row, counts[row[0]]) for row in rows) for rows in owners),
                   tuple(merges))
