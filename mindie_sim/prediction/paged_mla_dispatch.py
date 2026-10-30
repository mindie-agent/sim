"""Explicit paged MLA host-owner metadata and complete partition validation.

[Role] Preserve CPU host dispatch output separately from kernel tile mechanics.
[Invariants] AIC segments partition each sequence's real KV tiles exactly once.
  AIV ranges partition each split output's physical heads exactly once. Context
  identity is explicit; a declaration for one geometry cannot price another.
[Timeouts & Locks] O(owners + segments + batch), no tile expansion or locks.
[Side-Effect Policy] Pure immutable metadata only; no source compilation or I/O.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class Dispatch:
    owners: tuple
    merges: tuple

    @classmethod
    def parse(cls, raw, contexts, heads):
        if raw is None:
            return None
        if tuple(raw['contexts']) != contexts or raw['heads'] != heads:
            raise ModelingError('paged_mla_dispatch_geometry_mismatch')
        owners = tuple(tuple(tuple(span) for span in rows) for rows in raw['aic_owner_segments'])
        merges = tuple(tuple(span) for span in raw['aiv_merge_segments'])
        if not owners or any(not spans for spans in owners) or len(merges) > 2*len(owners):
            raise ModelingError('paged_mla_dispatch_owner_domain')
        kv = defaultdict(list)
        for spans in owners:
            for span in spans:
                if (len(span) != 3 or any(type(x) is not int for x in span)
                        or not 0 <= span[0] < len(contexts) or span[1] < 0 or span[2] < 1):
                    raise ModelingError('paged_mla_dispatch_invalid_kv_segment')
                kv[span[0]].append((span[1],span[1]+span[2]))
        cls._cover(kv, {i:(c+128)//128 for i,c in enumerate(contexts)}, 'kv')
        output = defaultdict(list)
        for span in merges:
            if (len(span) != 3 or any(type(x) is not int for x in span)
                    or span[0] not in kv or len(kv[span[0]]) <= 1
                    or span[1] < 0 or span[2] < 1):
                raise ModelingError('paged_mla_dispatch_invalid_fd_segment')
            output[span[0]].append((span[1],span[1]+span[2]))
        cls._cover(output, {i:heads for i,spans in kv.items() if len(spans)>1}, 'fd')
        return cls(owners,merges)

    @staticmethod
    def _cover(spans, sizes, kind):
        if set(spans) != set(sizes):
            raise ModelingError(f'paged_mla_dispatch_{kind}_coverage')
        for sequence, size in sizes.items():
            cursor = 0
            for start,end in sorted(spans[sequence]):
                if start != cursor:
                    raise ModelingError(f'paged_mla_dispatch_{kind}_overlap_or_gap')
                cursor = end
            if cursor != size:
                raise ModelingError(f'paged_mla_dispatch_{kind}_extent')

    def jobs(self):
        counts = Counter(span[0] for rows in self.owners for span in rows)
        return tuple(tuple((*span,counts[span[0]]) for span in rows) for rows in self.owners)
