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


SOURCE_REVISION = "a3925fd49cd256ad478d2faf983bf2961b8ae2c8"
SOURCE_PATH = ("attention/fused_infer_attention_score/op_host/"
               "flash_attention_infer_tiling.h")
HOST_KV_TILE = 512


def partitions(requests, query_heads, kv_heads, cores, *, key_tile, query_tile):
    """Return {(request, KV head): ((owner, micro_start, micro_end), ...)}.

    Requests are (query length, preceding context); host lengths include the
    current query. The selected FAI flash-decode branch has q <= 16 and fits a
    whole GQA query group in 128 rows. Unsupported host branches remain explicit
    instead of silently reusing this mapping for arbitrary attention.
    """
    group = query_heads // kv_heads
    lengths = tuple((q, context + q) for q, context in requests)
    if (cores is None or not 1 <= cores <= 26 or not lengths
            or key_tile > HOST_KV_TILE or HOST_KV_TILE % key_tile):
        raise ModelingError("fai_decode_mapping_requires_1_to_26_cores_and_512_divisor_tile")
    if any(not 0 < q <= 16 or q * group > min(128, query_tile)
           or min(max(1, (128 // q) // 2 * 2), group) != group
           for q, _ in lengths):
        raise ModelingError("fai_decode_mapping_requires_one_query_head_group_tile")
    longest = max(k for _, k in lengths)
    tasks = len(requests) * kv_heads
    if not (longest >= 1024 and (
            (5 * tasks <= 4 * cores and longest >= cores * HOST_KV_TILE)
            or 5 * tasks <= 2 * cores)):
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
