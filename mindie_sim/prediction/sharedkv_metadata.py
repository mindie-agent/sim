"""Source metadata partition for one SCFA sequence on the arch32 path.

[Role] Assign complete query rows to consecutive AIC owners.
[Invariants] The source's integer scheduling costs choose ownership only; they
  are never durations, service rates or physical work. S2 splitting is disabled.
[Timeouts & Locks] O(cores + cost bands) work/state; bounded pure-plan cache.
[Side-Effect Policy] Pure geometry arithmetic; no observations or device access.
"""
from functools import lru_cache


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
    m_cost = 6 * ((heads + 15) // 16)
    cost = sum(m_cost * ((rows + 511) // 512) + 10 * ((rows + 63) // 64)
               for rows in (original, selected))
    last_rows = ((selected or original) - 1) % 512 + 1
    return cost, m_cost + 10 * ((last_rows + 63) // 64)


def _cost_bands(query, context, window, ratio, topk, heads):
    # Every source cost transition is a 64-row boundary. The 512-row tile
    # boundary also belongs to this set, including its last-block reset.
    boundaries = {0, query}
    thresholds = (*range(65, window + 1, 64),
                  *(ratio * rows for rows in range(1, topk + 1, 64)))
    boundaries.update(position for threshold in thresholds
                      if 0 < (position := threshold - context - 1) < query)
    ordered = sorted(boundaries)
    return tuple((start, stop, *_row_cost(start, context, window, ratio, topk, heads))
                 for start, stop in zip(ordered, ordered[1:]))


@lru_cache(maxsize=32)
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
    last_tolerance = bands[-1][3] // 2
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
