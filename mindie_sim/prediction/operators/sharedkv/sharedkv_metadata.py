"""Source metadata partition for one SCFA sequence on the arch32 path.

[Role] Assign complete query rows to consecutive AIC owners.
[Invariants] The source's integer scheduling costs choose ownership only; they
  are never durations, service rates or physical work. S2 splitting is disabled.
[Timeouts & Locks] O(cores + cost bands) work/state; bounded pure-plan cache.
[Side-Effect Policy] Pure geometry arithmetic; no observations or device access.
"""
from functools import lru_cache


FD_KEY_TILE_ROWS = 512
FD_M_BLOCK_COST = 6
FD_M_BLOCK_ROWS = 16
FD_S2_BLOCK_COST = 10
FD_S2_BLOCK_ROWS = 64
LAST_TILE_COST_INDEX = 3
OWNER_PLAN_CACHE_ENTRIES = 32


OWNER_SCHEDULE = "metadata_cost_balanced"
SOURCE = {
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/",
    "cpp_sha256": "ba801a3c9904993e01efda8fc48117bfca6c1906b571ff0811238e4f88687b4f",
    "header_sha256": "f574419d3bc44d5612e8f13f828ebfc726c2148daa448a502676e2bb83ace189",
    "scope": "single sequence, one KV head, mBaseSize=groupSize, s2BaseSize=512; supportFd=false",
    "cost_basis": "Win/CmpCalcCost: 6*ceil(M/16)+10*ceil(S2/64); ownership weights only",
    "partition": "AssignByBatch/AssignByRow; limit=max(remaining/core_count,max_row_cost), tolerance=last_block_cost/2",
    "capture_identity": "not asserted",
}


def _row_cost(position, context, window, ratio, topk, heads):
    original = min(context + position + 1, window)
    selected = min((context + position + 1) // ratio, topk)
    m_cost = FD_M_BLOCK_COST * ((heads + (FD_M_BLOCK_ROWS - 1)) // FD_M_BLOCK_ROWS)
    cost = sum(m_cost * ((rows + (FD_KEY_TILE_ROWS - 1)) // FD_KEY_TILE_ROWS) + FD_S2_BLOCK_COST * ((rows + (FD_S2_BLOCK_ROWS - 1)) // FD_S2_BLOCK_ROWS)
               for rows in (original, selected))
    last_rows = ((selected or original) - 1) % FD_KEY_TILE_ROWS + 1
    return cost, m_cost + FD_S2_BLOCK_COST * ((last_rows + (FD_S2_BLOCK_ROWS - 1)) // FD_S2_BLOCK_ROWS)


def _cost_bands(query, context, window, ratio, topk, heads):
    # Every source cost transition is a 64-row boundary. The 512-row tile
    # boundary also belongs to this set, including its last-block reset.
    boundaries = {0, query}
    thresholds = (*range((FD_S2_BLOCK_ROWS + 1), window + 1, FD_S2_BLOCK_ROWS),
                  *(ratio * rows for rows in range(1, topk + 1, FD_S2_BLOCK_ROWS)))
    boundaries.update(position for threshold in thresholds
                      if 0 < (position := threshold - context - 1) < query)
    ordered = sorted(boundaries)
    return tuple((start, stop, *_row_cost(start, context, window, ratio, topk, heads))
                 for start, stop in zip(ordered, ordered[1:]))


@lru_cache(maxsize=OWNER_PLAN_CACHE_ENTRIES)
def query_partitions(query, context, window, ratio, topk, heads, compute_units):
    """Return enabled owners' half-open query ranges, matching source metadata.

    With one sequence and no S2 split, the source first accepts the remaining
    batch if it fits. Otherwise it consumes whole rows while each next row fits
    the core limit plus half that row's last S2 block cost. Equal-cost runs can
    be consumed arithmetically without changing those integer comparisons.
    """
    if query == 0:
        return ()
    bands = _cost_bands(query, context, window, ratio, topk, heads)
    remaining = sum((stop - start) * cost for start, stop, cost, _ in bands)
    maximum = max(cost for _, _, cost, _ in bands)
    last_tolerance = bands[-1][LAST_TILE_COST_INDEX] // 2
    cursor = band_index = 0
    partitions = []
    for core in range(compute_units):
        if cursor == query:
            break
        limit = max(remaining // (compute_units - core), maximum)
        start = cursor
        if remaining <= limit + last_tolerance:
            partitions.append((start, query))
            break
        spent = 0
        while band_index < len(bands):
            _, stop, cost, last = bands[band_index]
            count = min(stop - cursor, max(0, (limit + last // 2 - spent) // cost))
            cursor += count
            spent += count * cost
            if cursor < stop:
                break
            band_index += 1
        partitions.append((start, cursor))
        remaining -= spent
    return tuple(partitions)
