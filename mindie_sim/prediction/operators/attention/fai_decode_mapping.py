"""FAInfer flash-decode host partitions for a staged analytical GQA body.

[Role] Reproduce the small-query, paged TND FAI host's weighted KV partition.
[Invariants] Owners consume consecutive 512-row blocks across request/head
  boundaries. The budget resets per owner; a block's excess is not carried to
  the next owner. KV tails and ragged lengths remain physical work. This maps
  owners only: it does not identify a captured binary or price its device code.
[Timeouts & Locks] O(requests * KV heads + owners); no per-KV-token expansion.
[Side-Effect Policy] Pure arithmetic, no measurement lookup or external I/O.
"""
from mindie_sim.schema.common import ModelingError


HOST_CORE_RECORD_LIMIT = 64
HOST_MAX_QUERY_TOKENS = 16
HOST_QUERY_ROW_LIMIT = 128
OCCUPANCY_RATIO_DENOMINATOR = 5
SHORT_KV_OCCUPANCY_NUMERATOR = 4


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


def partitions(requests, query_heads, kv_heads, cores, *, key_tile, query_tile):
    """Return {(request, KV head): ((owner, micro_start, micro_end), ...)}.

    Requests are (query length, preceding context); host lengths include the
    current query. The selected FAI flash-decode branch has q <= 16 and fits a
    whole GQA query group in 128 rows. Unsupported host branches remain explicit
    instead of silently reusing this mapping for arbitrary attention.
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

    total = sum(q * k for q, k in lengths) * query_heads
    budget = (total + cores - 1) // cores
    owner, remaining = 0, budget
    micro_per_host = HOST_KV_TILE // key_tile
    result = {}
    for request, (query, length) in enumerate(lengths):
        host_tiles = (length + HOST_KV_TILE - 1) // HOST_KV_TILE
        row_work = query * group
        for head in range(kv_heads):
            ranges, start = [], 0
            while start < host_tiles:
                full_tile_work = row_work * HOST_KV_TILE
                count = min(host_tiles - start,
                            (remaining + full_tile_work - 1) // full_tile_work)
                end = start + count
                ranges.append((owner, start * micro_per_host,
                               min(end * micro_per_host,
                                   (length + key_tile - 1) // key_tile)))
                remaining -= row_work * (min(end * HOST_KV_TILE, length)
                                         - start * HOST_KV_TILE)
                start = end
                if remaining <= 0:
                    owner += 1
                    remaining = budget
            result[request, head] = tuple(ranges)
    return result
