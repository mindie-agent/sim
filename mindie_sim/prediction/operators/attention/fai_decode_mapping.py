"""FAInfer flash-decode host partitions for a staged analytical GQA body.

[Role] Reproduce the small-query, paged TND FAI host's weighted KV partition.
[Invariants] Owners consume consecutive 512-row blocks across request/head
  boundaries. The budget resets per owner; a block's excess is not carried to
  the next owner. KV tails and ragged lengths remain physical work. This maps
  owners only: it does not identify a captured binary or price its device code.
[Timeouts & Locks] O(requests * KV heads + owners); no per-KV-token expansion.
[Side-Effect Policy] Pure arithmetic, no measurement lookup or external I/O.
"""
import math

from mindie_sim.schema.common import ModelingError


HOST_CORE_RECORD_LIMIT = 64
MAX_CORE_NUM_FD = 26
HOST_MAX_QUERY_TOKENS = 16
HOST_QUERY_ROW_LIMIT = 128
OCCUPANCY_RATIO_DENOMINATOR = 5
SHORT_KV_OCCUPANCY_NUMERATOR = 4

# CANN flash_attention_infer_tiling.h:112-117 aligned block-cost constants.
FD_COST_ALIGN_M = 16
FD_COST_ALIGN_S2 = 64
FD_COST_COEF_M = 10
FD_COST_COEF_S2 = 10
FD_TOLERANCE_RATIO = 2
FD_CORE_LAUNCH_COST = 36


SOURCE_REVISION = "a3925fd49cd256ad478d2faf983bf2961b8ae2c8"
SOURCE_PATH = ("attention/fused_infer_attention_score/op_host/"
               "flash_attention_infer_tiling.h")
HOST_KV_TILE = 512
ARCH35_FLASH_SPLIT_MIN_KV_LEN = 1024
ARCH35_FLASH_SPLIT_MAX_KV_LEN = 65536
ARCH35_FLASH_SPLIT_MAX_HEAD_DIM = 128
ARCH35_MIN_TOKENS_PER_SPLIT_CORE = 1024
ARCH35_FLASH_SPLIT_CORE_DIVISOR = 4
MAX_SUPPORTED_HOST_CORES = 64
MAX_DECODE_QUERY_LEN = 16
QUERY_GROUP_TILE_ROWS = 128


def uses_flash_decode(requests, kv_heads, cores):
    """Pinned host selector for the supported paged, small-query domain."""
    longest = max(context + q for q, context in requests)
    tasks = len(requests) * kv_heads
    return longest >= 1024 and (
        (OCCUPANCY_RATIO_DENOMINATOR * tasks <= SHORT_KV_OCCUPANCY_NUMERATOR * cores and longest >= cores * HOST_KV_TILE)
        or OCCUPANCY_RATIO_DENOMINATOR * tasks <= 2 * cores)


def uses_flash_decode_arch35(
    requests, kv_heads, cores, *, head_dim=ARCH35_FLASH_SPLIT_MAX_HEAD_DIM
):
    """CANN 9.3 Arch35 FIA flash-decode KV-split selector (hd <= 128, 1K <= L < 64K)."""
    longest = max(context + q for q, context in requests)
    tasks = len(requests) * kv_heads
    if (
        head_dim > ARCH35_FLASH_SPLIT_MAX_HEAD_DIM
        or longest < ARCH35_FLASH_SPLIT_MIN_KV_LEN
        or longest >= ARCH35_FLASH_SPLIT_MAX_KV_LEN
        or tasks * ARCH35_FLASH_SPLIT_CORE_DIVISOR > cores
    ):
        return False
    return uses_flash_decode(requests, kv_heads, cores)


def _validate_host_domain(requests, query_heads, kv_heads, cores, *, key_tile, query_tile):
    group = query_heads // kv_heads
    if (
        cores is None
        or not 1 <= cores <= MAX_SUPPORTED_HOST_CORES
        or not requests
        or key_tile > HOST_KV_TILE
        or HOST_KV_TILE % key_tile
        or any(
            not 0 < q <= MAX_DECODE_QUERY_LEN
            or q * group > min(QUERY_GROUP_TILE_ROWS, query_tile)
            or min(max(1, (QUERY_GROUP_TILE_ROWS // q) // 2 * 2), group) != group
            for q, _ in requests
        )
    ):
        raise ModelingError("fai_host_mapping_requires_supported_single_query_head_tile")


def _regular_host_partitions(requests, query_heads, kv_heads, cores, *, key_tile):
    group = query_heads // kv_heads
    if all(q == 1 for q, _ in requests) and len(requests) >= cores:
        # SplitCoreDecodeBS1GN2 packs up to kv_block_tile KV heads per batch task,
        # then unpacks tail tasks into single-KV-head tasks when 0 < tail <= cores // 2.
        kv_block_tile = min(max(1, QUERY_GROUP_TILE_ROWS // group), kv_heads)
        tasks_per_batch = (kv_heads + kv_block_tile - 1) // kv_block_tile
        total_tasks = len(requests) * tasks_per_batch
        tail_tasks = total_tasks % cores
        rework = 0 < tail_tasks <= cores // 2 and kv_block_tile > 1
        packed_limit = total_tasks - tail_tasks if rework else total_tasks
        result = {}
        task_idx = 0
        rework_idx = packed_limit
        for batch, (query, context) in enumerate(requests):
            tiles = (context + query + key_tile - 1) // key_tile
            for block_idx in range(tasks_per_batch):
                h_start = block_idx * kv_block_tile
                h_end = min(kv_heads, h_start + kv_block_tile)
                if task_idx < packed_limit:
                    owner = task_idx % cores
                    for head in range(h_start, h_end):
                        result[(batch, head)] = ((owner, 0, tiles),)
                else:
                    for head in range(h_start, h_end):
                        result[(batch, head)] = ((rework_idx % cores, 0, tiles),)
                        rework_idx += 1
                task_idx += 1
        return result
    return {
        (batch, head): (
            ((batch * kv_heads + head) % cores, 0, (context + query + key_tile - 1) // key_tile),
        )
        for batch, (query, context) in enumerate(requests)
        for head in range(kv_heads)
    }


def host_partitions(requests, query_heads, kv_heads, cores, *, key_tile, query_tile):
    """Select flash splitting or regular round-robin query/head tasks."""
    _validate_host_domain(
        requests, query_heads, kv_heads, cores, key_tile=key_tile, query_tile=query_tile
    )
    if uses_flash_decode(requests, kv_heads, cores):
        return partitions(
            requests, query_heads, kv_heads, cores, key_tile=key_tile, query_tile=query_tile
        )
    return _regular_host_partitions(
        requests, query_heads, kv_heads, cores, key_tile=key_tile
    )


def host_partitions_arch35(
    requests,
    query_heads,
    kv_heads,
    cores,
    *,
    key_tile,
    query_tile,
    head_dim=ARCH35_FLASH_SPLIT_MAX_HEAD_DIM,
):
    """CANN 9.3 Arch35 FIA host partitioner (splits hd<=128 in [1K, 64K), regular above)."""
    _validate_host_domain(
        requests, query_heads, kv_heads, cores, key_tile=key_tile, query_tile=query_tile
    )
    if uses_flash_decode_arch35(requests, kv_heads, cores, head_dim=head_dim):
        longest = max(context + q for q, context in requests)
        split_cores = min(
            cores,
            max(kv_heads, longest // ARCH35_MIN_TOKENS_PER_SPLIT_CORE),
        )
        if uses_flash_decode(requests, kv_heads, split_cores):
            return partitions(
                requests,
                query_heads,
                kv_heads,
                split_cores,
                key_tile=key_tile,
                query_tile=query_tile,
            )
    return _regular_host_partitions(
        requests, query_heads, kv_heads, cores, key_tile=key_tile
    )


def _fd_block_cost(m_rows: int, s2_len: int) -> int:
    """Weighted cost of one (M-row, S2-token) block per CalcFdBlockCost."""
    align_m = (m_rows + FD_COST_ALIGN_M - 1) // FD_COST_ALIGN_M
    align_s2 = (s2_len + FD_COST_ALIGN_S2 - 1) // FD_COST_ALIGN_S2
    return FD_COST_COEF_M * align_m + FD_COST_COEF_S2 * align_s2


def _consume_s2_blocks(
    blocks: int, full_cost: int, tail_cost: int, s2_idx: int, remaining: int,
) -> tuple[int, int]:
    """O(1) equivalent of ConsumeS2BlocksFD across full 512-row blocks and tail."""
    full_blocks = blocks - 1
    if s2_idx < full_blocks and remaining > 0:
        take = min(full_blocks - s2_idx, (remaining + full_cost - 1) // full_cost)
        remaining -= take * full_cost
        s2_idx += take
    if s2_idx == full_blocks and remaining > 0:
        remaining -= tail_cost
        s2_idx += 1
    return s2_idx, remaining


def _append_cursor_slices(
    buckets: dict[tuple[int, int], list[tuple[int, int, int]]],
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    *,
    batch_count: int,
    kv_heads: int,
    blocks_by_req: tuple[int, ...],
    micro_tiles: tuple[int, ...],
    micro_per_host: int,
    owner: int,
) -> None:
    """Emit micro-tile intervals for [start, end) in (batch, kv_head, host_s2) order."""
    b_idx, n1_idx, s2_lo = start
    end_b, end_n1, end_s2 = end
    while b_idx < batch_count and (b_idx, n1_idx) <= (end_b, end_n1):
        at_end = b_idx == end_b and n1_idx == end_n1
        s2_hi = end_s2 if at_end else blocks_by_req[b_idx]
        if s2_hi > s2_lo:
            buckets[b_idx, n1_idx].append((
                owner,
                s2_lo * micro_per_host,
                min(s2_hi * micro_per_host, micro_tiles[b_idx]),
            ))
        if at_end:
            break
        n1_idx += 1
        s2_lo = 0
        if n1_idx == kv_heads:
            b_idx += 1
            n1_idx = 0


def partitions(requests, query_heads, kv_heads, cores, *, key_tile, query_tile):
    """Return {(request, KV head): ((owner, micro_start, micro_end), ...)}.

    Requests are (query length, preceding context); host lengths include the
    current query. Active core selection and cost-balanced cursor assignment
    follow CANN ``splitBN2S1GS2`` / ``fillCoreInfoForFlashDecode`` in
    ``flash_attention_infer_tiling.h``.
    """
    group = query_heads // kv_heads
    lengths = tuple((q, context + q) for q, context in requests)
    if (cores is None or not 1 <= cores <= HOST_CORE_RECORD_LIMIT or not lengths
            or key_tile > HOST_KV_TILE or HOST_KV_TILE % key_tile):
        raise ModelingError("fai_decode_mapping_requires_1_to_26_cores_and_512_divisor_tile")
    if any(not 0 < q <= HOST_MAX_QUERY_TOKENS or q * group > min(HOST_QUERY_ROW_LIMIT, query_tile)
           or min(max(1, (HOST_QUERY_ROW_LIMIT // q) // 2 * 2), group) != group
           for q, _ in lengths):
        raise ModelingError("fai_decode_mapping_requires_one_query_head_group_tile")
    if not uses_flash_decode(requests, kv_heads, cores):
        raise ModelingError("fai_decode_mapping_outside_flash_decode_host_domain")

    blocks_by_req = tuple((length + HOST_KV_TILE - 1) // HOST_KV_TILE for _, length in lengths)
    micro_tiles = tuple((length + key_tile - 1) // key_tile for _, length in lengths)
    full_costs = tuple(_fd_block_cost(q * group, HOST_KV_TILE) for q, _ in lengths)
    tail_costs = tuple(
        _fd_block_cost(q * group, length - (blocks - 1) * HOST_KV_TILE)
        for (q, length), blocks in zip(lengths, blocks_by_req)
    )
    n1_costs = tuple(
        (blocks - 1) * full_cost + tail_cost
        for blocks, full_cost, tail_cost in zip(blocks_by_req, full_costs, tail_costs)
    )
    tolerances = tuple(full_cost // FD_TOLERANCE_RATIO for full_cost in full_costs)
    total_cost = sum(kv_heads * cost for cost in n1_costs)
    total_blocks = sum(kv_heads * blocks for blocks in blocks_by_req)
    max_core = min(cores, total_blocks, MAX_CORE_NUM_FD)
    best_core = max(
        1,
        min(max_core, int(math.sqrt(total_cost / float(FD_CORE_LAUNCH_COST)) + 0.5)),
    )

    batch_count = len(lengths)
    micro_per_host = HOST_KV_TILE // key_tile
    buckets: dict[tuple[int, int], list[tuple[int, int, int]]] = {
        (req, head): [] for req in range(batch_count) for head in range(kv_heads)
    }
    now_b = now_n1 = now_s2 = 0
    remaining_cost = total_cost

    def advance(b_idx: int, n1_idx: int, s2_idx: int) -> tuple[int, int, int]:
        if b_idx < batch_count and s2_idx == blocks_by_req[b_idx]:
            n1_idx += 1
            s2_idx = 0
        if b_idx < batch_count and n1_idx == kv_heads:
            b_idx += 1
            n1_idx = 0
            s2_idx = 0
        return b_idx, n1_idx, s2_idx

    for core_idx in range(best_core):
        if now_b >= batch_count:
            break
        cost_limit = remaining_cost // (best_core - core_idx)
        res = cost_limit
        start = (now_b, now_n1, now_s2)

        # Stage 1: consume S2 blocks of the current (batch, KV head).
        now_s2, res = _consume_s2_blocks(
            blocks_by_req[now_b], full_costs[now_b], tail_costs[now_b], now_s2, res,
        )
        if res <= 0:
            end = (now_b, now_n1, now_s2)
        else:
            now_b, now_n1, now_s2 = advance(now_b, now_n1, now_s2)
            # Stage 2: jump whole remaining batches within half-block tolerance.
            while now_b < batch_count and res > 0:
                rest_batch = (kv_heads - now_n1) * n1_costs[now_b]
                if res + tolerances[now_b] >= rest_batch:
                    res -= rest_batch
                    now_b += 1
                    now_n1 = now_s2 = 0
                else:
                    break
            # Stage 3: jump whole remaining KV heads within tolerance.
            while now_b < batch_count and now_n1 < kv_heads and res > 0:
                if res + tolerances[now_b] >= n1_costs[now_b]:
                    res -= n1_costs[now_b]
                    now_n1 += 1
                    now_s2 = 0
                else:
                    break
            now_b, now_n1, now_s2 = advance(now_b, now_n1, now_s2)
            # Stage 5: consume S2 blocks to exact landing point.
            if now_b < batch_count and res > 0:
                now_s2, res = _consume_s2_blocks(
                    blocks_by_req[now_b], full_costs[now_b], tail_costs[now_b], now_s2, res,
                )
            end = (now_b, now_n1, now_s2)

        _append_cursor_slices(
            buckets, start, end, batch_count=batch_count, kv_heads=kv_heads,
            blocks_by_req=blocks_by_req, micro_tiles=micro_tiles,
            micro_per_host=micro_per_host, owner=core_idx,
        )
        now_b, now_n1, now_s2 = advance(*end)
        consumed = max(0, cost_limit - res)
        remaining_cost = max(0, remaining_cost - consumed)

    return {key: tuple(spans) for key, spans in buckets.items()}
