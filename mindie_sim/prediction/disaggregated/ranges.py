"""Physical cache extents for the P/D connector.

[Role] Project cache owners onto global component rows and packed local storage.
[Invariants] Every delivered row maps to one published source row.
[Timeouts & Locks] Finite interval sweeps, no locks.
[Side-Effect Policy] Address reservations live only for the current handoff.
"""
from mindie_sim.catalog.cache import index_k_block_rows
from mindie_sim.catalog.paired_sfa import execution
from mindie_sim.schema.common import ModelingError


def address(bases, extents, storage, plane, size, displacement, count):
    key = (storage, plane)
    if displacement < 0 or displacement + count > size:
        raise ModelingError("pd_transfer_outside_source_plane")
    if key not in bases:
        bases[key] = extents[storage]
        extents[storage] += size
    return bases[key] + displacement


def owned_runs(component, layer, model, parallel, rank, span, tokens, window_start):
    lo, hi = span
    if component in {"mla_kv", "window"}:
        return ((max(lo, window_start), hi),) if hi > max(lo, window_start) else ()
    ratio = max(1, layer.compress_ratio)
    if component == "compressed_kv":
        return ((lo // ratio, hi // ratio),) if hi // ratio > lo // ratio else ()
    ratio = max(ratio, layer.index_kpool)
    total = tokens // ratio
    if execution(model).get("dcp_indexer_sharded") and parallel.cp > 1:
        return ((lo // ratio, hi // ratio),) if hi // ratio > lo // ratio else ()
    if parallel.cp_plan != "dsa_cp" or parallel.cp == 1:
        return ((0, total),) if total else ()
    block = index_k_block_rows(ratio)
    start = (rank % parallel.cp) * block
    return tuple((i, min(total, i + block))
                 for i in range(start, total, parallel.cp * block))


def transfers_for_runs(sources, wanted):
    """Stream adjacent pieces. Replicas tie-break in placement order."""
    from bisect import bisect_right
    ranks = []
    for rank, runs, size in sources:
        rows = sum(hi - lo for lo, hi in runs)
        if rows and size:
            prefix, offsets = 0, []
            for lo, hi in runs:
                offsets.append(prefix)
                prefix += hi - lo
            ranks.append((rank, runs, tuple(lo for lo, _ in runs),
                          offsets, size, size / rows))
    for begin, stop in wanted:
        pos = begin
        while pos < stop:
            for rank, runs, starts, offsets, size, unit in ranks:
                i = bisect_right(starts, pos) - 1
                if i >= 0 and pos < runs[i][1]:
                    end = min(stop, runs[i][1])
                    yield (rank, pos, end, size, (offsets[i] + pos - starts[i]) * unit, unit)
                    pos = end
                    break
            else:
                raise ModelingError(f"pd_handoff_token_range_uncovered:{pos}:{stop}")
