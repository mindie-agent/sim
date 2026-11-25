"""Physical cache extents for the P/D connector.

[Role] Project cache owners onto global component rows and packed local storage.
[Invariants] Every delivered row maps to one published source row with exact integer
  half-open ``[start, end)`` byte intervals.
[Timeouts & Locks] Finite interval sweeps, no locks.
[Side-Effect Policy] Address reservations live only for the current handoff.
"""
from bisect import bisect_right
import math

from mindie_sim.catalog.cache import index_k_block_rows
from mindie_sim.catalog.context_parallel import dcp_indexer_sharded
from mindie_sim.schema.common import ModelingError


def exact_int_bytes(value, label: str) -> int:
    """Validate and convert a non-negative integer byte count (rejecting fractional floats)."""
    if isinstance(value, bool):
        raise ModelingError(f"{label}_invalid:{value}")
    if isinstance(value, int):
        if value < 0:
            raise ModelingError(f"{label}_invalid:{value}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            raise ModelingError(f"{label}_requires_exact_int:{value}")
        return int(value)
    raise ModelingError(f"{label}_invalid:{value}")


def address(bases, extents, storage, plane, size, displacement, count):
    """Reserve a contiguous integer byte plane and return the displaced integer offset."""
    for label, val in (("size", size), ("displacement", displacement), ("count", count)):
        if isinstance(val, bool) or not isinstance(val, int):
            raise ModelingError(f"pd_transfer_address_requires_int:{label}:{val}")
    key = (storage, plane)
    if displacement < 0 or count < 0 or displacement + count > size:
        raise ModelingError("pd_transfer_outside_source_plane")
    if key not in bases:
        bases[key] = int(extents[storage])
        extents[storage] = int(extents[storage]) + size
    return bases[key] + displacement


def _compressed_store_on_completion_runs(
    lo: int, hi: int, tokens: int, ratio: int,
) -> tuple[tuple[int, int], ...]:
    """Exact half-open compressed row range ``[j_lo, j_hi)`` closing inside ``[lo, hi)``.

    Row ``j >= 0`` closes at raw token ``(j + 1) * ratio - 1``. Thus
    ``lo <= (j + 1) * ratio - 1 < hi`` holds iff ``lo // ratio <= j < hi // ratio``.
    """
    total = max(0, int(tokens)) // ratio
    if total <= 0 or hi <= lo:
        return ()
    c_lo = max(0, int(lo)) // ratio
    c_hi = min(total, max(0, int(hi)) // ratio)
    return ((c_lo, c_hi),) if c_hi > c_lo else ()


def owned_runs(component, layer, model, parallel, rank, span, tokens, window_start,
               *, runtime=None):
    """Exact integer half-open row intervals owned by ``rank`` for ``component``."""
    lo, hi = int(span[0]), int(span[1])
    toks = max(0, int(tokens))
    win_lo = max(0, int(window_start))
    if component in {"mla_kv", "window"}:
        start = max(lo, win_lo)
        return ((start, hi),) if hi > start else ()
    ratio = max(1, int(getattr(layer, "compress_ratio", 1) or 1))
    if component == "compressed_kv":
        return _compressed_store_on_completion_runs(lo, hi, toks, ratio)
    ratio = max(ratio, int(getattr(layer, "index_kpool", 1) or 1))
    total = toks // ratio
    if dcp_indexer_sharded(model, runtime) and int(parallel.cp) > 1:
        return _compressed_store_on_completion_runs(lo, hi, toks, ratio)
    if parallel.cp_plan != "dsa_cp" or int(parallel.cp) <= 1:
        return ((0, total),) if total > 0 else ()
    block = index_k_block_rows(ratio)
    cp = int(parallel.cp)
    start = (int(rank) % cp) * block
    return tuple((i, min(total, i + block))
                 for i in range(start, total, cp * block))


def transfers_for_runs(sources, wanted):
    """Stream adjacent integer byte-interval pieces. Replicas tie-break in source order."""
    ranks = []
    for rank, runs, raw_size in sources:
        int_runs = tuple((int(lo), int(hi)) for lo, hi in runs if int(hi) > int(lo))
        rows = sum(hi - lo for lo, hi in int_runs)
        size = exact_int_bytes(raw_size, "pd_transfer_source_size")
        if rows <= 0 or size <= 0:
            continue
        if size % rows != 0:
            raise ModelingError(
                f"pd_transfer_non_integer_row_bytes:rank={rank},size={size},rows={rows}"
            )
        unit = size // rows
        prefix, offsets = 0, []
        for lo, hi in int_runs:
            offsets.append(prefix)
            prefix += hi - lo
        ranks.append((int(rank), int_runs, tuple(lo for lo, _ in int_runs),
                      tuple(offsets), size, unit))
    for begin, stop in wanted:
        pos = int(begin)
        stop_i = int(stop)
        while pos < stop_i:
            for rank, runs, starts, offsets, size, unit in ranks:
                i = bisect_right(starts, pos) - 1
                if i >= 0 and pos < runs[i][1]:
                    end = min(stop_i, runs[i][1])
                    offset = (offsets[i] + pos - starts[i]) * unit
                    yield (rank, pos, end, size, offset, unit)
                    pos = end
                    break
            else:
                raise ModelingError(f"pd_handoff_token_range_uncovered:{pos}:{stop_i}")
