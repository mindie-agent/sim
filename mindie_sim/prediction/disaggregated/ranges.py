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
from mindie_sim.catalog.context_parallel import (
    cp_interleave,
    dcp_indexer_sharded,
    layer_dcp_degree,
    layer_dcp_factors,
)
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


def canonical_physical_rank(
    index: int, parallel, *, stage: int = 0, replica: int = 0, cohort_size: int | None = None,
) -> int:
    """Canonical global physical rank ``((dp_i * pp + pp_i) * pcp + pcp_i) * tp + tp_i``."""
    tp_dim = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp_dim = max(1, int(getattr(parallel, "pcp", 1) or 1))
    pp_dim = max(1, int(getattr(parallel, "pp", 1) or 1))
    dcp_dp = max(1, int(getattr(parallel, "dcp_dp_width", 1) or 1))
    idx = int(index)
    tp_i = idx % tp_dim
    pcp_i = (idx // tp_dim) % pcp_dim
    dp_local = idx // (tp_dim * pcp_dim)
    local_stage_w = tp_dim * pcp_dim
    if dcp_dp > 1 and cohort_size is not None and int(cohort_size) >= local_stage_w * dcp_dp:
        dp_i = (int(replica) // dcp_dp) * dcp_dp + dp_local
    else:
        dp_i = int(replica) + dp_local
    return ((dp_i * pp_dim + int(stage)) * pcp_dim + pcp_i) * tp_dim + tp_i


def dcp_cp_rank(layer, parallel, rank: int) -> int:
    """Exact DCP shard index for canonical global rank ``rank`` matching ``catalog.cache``."""
    cp = layer_dcp_degree(layer, parallel)
    if cp <= 1:
        return 0
    pcp_f, tp_g, dp_w = layer_dcp_factors(layer, parallel)
    tp_dim = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp_dim = max(1, int(getattr(parallel, "pcp", 1) or 1))
    pp_dim = max(1, int(getattr(parallel, "pp", 1) or 1))
    dp_dim = max(1, int(getattr(parallel, "dp", 1) or 1))
    r_int = int(rank)
    tp_i = r_int % tp_dim
    pcp_i = (r_int // tp_dim) % pcp_dim
    dp_i = (r_int // (tp_dim * pcp_dim * pp_dim)) % dp_dim
    if dp_w > 1:
        return ((dp_i % dp_w) * (pcp_f * tp_g) + (pcp_i % pcp_f) * tp_g + (tp_i % tp_g)) % cp
    return ((tp_i % tp_g) * pcp_f + (pcp_i % pcp_f)) % cp


def interleaved_token_runs(
    tokens: int, cp: int, interleave: int, cp_rank: int, *, window_start: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Exact half-open interleaved token runs owned by DCP shard ``cp_rank`` in ``[window_start, tokens)``."""
    win_lo = max(0, int(window_start))
    hi_tok = max(0, int(tokens))
    if hi_tok <= win_lo:
        return ()
    cp_n = max(1, int(cp))
    if cp_n <= 1:
        return ((win_lo, hi_tok),)
    step = max(1, int(interleave))
    stride = cp_n * step
    start = (int(cp_rank) % cp_n) * step
    if start + step <= win_lo:
        skip = (win_lo - start) // stride
        start += skip * stride
        if start + step <= win_lo:
            start += stride
    runs: list[tuple[int, int]] = []
    for pos in range(start, hi_tok, stride):
        r_lo = max(win_lo, pos)
        r_hi = min(hi_tok, pos + step)
        if r_hi > r_lo:
            runs.append((r_lo, r_hi))
    return tuple(runs)


def owned_runs(component, layer, model, parallel, rank, span, tokens, window_start,
               *, runtime=None, explicit_span: bool = False):
    """Exact integer half-open row intervals owned by ``rank`` for ``component``."""
    lo, hi = int(span[0]), int(span[1])
    toks = max(0, int(tokens))
    win_lo = max(0, int(window_start))
    cp_deg = layer_dcp_degree(layer, parallel)
    use_interleaved = (not explicit_span) and cp_deg > 1
    if component in {"gqa_kv", "mla_kv", "window"}:
        if use_interleaved:
            return interleaved_token_runs(
                toks, cp_deg, cp_interleave(runtime), dcp_cp_rank(layer, parallel, int(rank)),
                window_start=win_lo,
            )
        start = max(lo, win_lo)
        end = min(toks, hi)
        return ((start, end),) if end > start else ()
    ratio = max(1, int(getattr(layer, "compress_ratio", 1) or 1))
    if component == "compressed_kv":
        if use_interleaved:
            total = toks // ratio
            stride = max(1, int(cp_interleave(runtime)) // ratio)
            return interleaved_token_runs(
                total, cp_deg, stride, dcp_cp_rank(layer, parallel, int(rank)),
            )
        return _compressed_store_on_completion_runs(lo, hi, toks, ratio)
    ratio = max(ratio, int(getattr(layer, "index_kpool", 1) or 1))
    total = toks // ratio
    if explicit_span and (dcp_indexer_sharded(model, runtime) or cp_deg <= 1):
        return _compressed_store_on_completion_runs(lo, hi, toks, ratio)
    if dcp_indexer_sharded(model, runtime) and cp_deg > 1:
        stride = max(1, int(cp_interleave(runtime)) // ratio)
        return interleaved_token_runs(
            total, cp_deg, stride, dcp_cp_rank(layer, parallel, int(rank)),
        )
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
