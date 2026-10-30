"""Source SplitCoreV2 ownership for q1 dense, unmasked paged MLA.

[Role] Select AIC KV segments and AIV reduction heads from actual geometry.
[Invariants] Integer dimension scores select ownership, never elapsed service.
  Zero-KV graph padding changes storage/endpoints, not active attention work.
[Timeouts & Locks] O(batch + owners) storage; bounded physical-core search.
[Side-Effect Policy] Pure metadata; no observations, compiler or device access.
"""
from collections import Counter
from math import isqrt

POLICY = 'source_split_core_v2'
SOURCE = {
    'repository': 'cann/ops-transformer',
    'revision': 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8',
    'path': 'attention/common/op_host/split_core_v2.cpp',
    'sha256': '2281771228e15ce82b6881417679a57e0dacc69662f5e51217588041809fc78d',
    'caller': 'attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp',
    'caller_lines': '297-309,359-361',
    'domain': 'q1; N2=1; heads<=64; no mask; M64/KV128; fdTolerance=9',
}


def source_dispatch(contexts, heads, cores, capacity=None):
    """Specialize the original batch/row/block assignment to one M row.

    q1 with <=64 heads has one M tile per sequence, so whole-batch and
    whole-row assignment coincide. No tensor or tile list is materialized.
    The source's no-FD alternative and FD head partition remain independent.
    """
    sizes = tuple((c + 128) // 128 for c in contexts)
    normal = 6 * ((heads + 15) // 16) + 20
    tails = tuple(6 * ((heads + 15) // 16) + 10 * (((c % 128) + 64) // 64)
                  for c in contexts)
    costs = tuple((n - 1) * normal + tail for n, tail in zip(sizes, tails))
    total = sum(costs)

    def assign(count, stream_k):
        seq = tile = 0
        remaining, maximum, ends = total, 0, []
        for owner in range(count):
            if seq == len(sizes):
                break
            limit = remaining // (count - owner)
            if not stream_k:
                limit = max(limit, max(costs))
            assigned = blocks = 0
            while seq < len(sizes):
                whole = costs[seq] - tile * normal
                if assigned + whole > limit + tails[seq] // 2:
                    break
                assigned += whole
                blocks += sizes[seq] - tile
                seq, tile = seq + 1, 0
            if seq < len(sizes) and stream_k:
                current = tails[seq] if tile == sizes[seq] - 1 else normal
                take = max(0, (limit + current // 2 - assigned) // current)
                assigned += take * current
                blocks += take
                tile += take
                if blocks == 0:
                    assigned, tile = current, tile + 1
                    if tile == sizes[seq]:
                        seq, tile = seq + 1, 0
            remaining -= assigned
            maximum = max(maximum, assigned)
            ends.append([seq, 0, tile])
        return ends, maximum

    maximum = min(cores, sum(sizes))
    minimum = min((isqrt(4 * sum(sizes) + 1) + 1) // 2, maximum)
    best, best_cost = None, None
    for count in range(minimum, maximum + 1):
        ends, cost = assign(count, True)
        if best_cost is None or cost < best_cost:
            best, best_cost = ends, cost

    def owners(ends):
        previous, result = (0, 0), []
        for seq, _, tile in ends:
            rows = []
            for index in range(previous[0], min(seq + 1, len(sizes))):
                start = previous[1] if index == previous[0] else 0
                end = tile if index == seq else sizes[index]
                if end > start:
                    rows.append([index, start, end - start])
            result.append(rows)
            previous = seq, tile
        return result

    assigned = owners(best)
    counts = Counter(row[0] for rows in assigned for row in rows)
    if any(n > 1 for n in counts.values()):
        no_fd, no_fd_cost = assign(len(best), False)
        # CheckChooseWithFd uses the actual M row, not the capacity M64.
        if no_fd_cost - 9 * normal <= best_cost:
            best, assigned = no_fd, owners(no_fd)
            counts = Counter(row[0] for rows in assigned for row in rows)

    fd = [{'sequence': seq, 'm_tile': 0, 'splits': n}
          for seq, n in sorted(counts.items()) if n > 1]
    fd_aiv, merge = [], []
    if fd:
        vectors = 2 * len(best)
        average = (sum(row['splits'] * heads for row in fd) + vectors - 1) // vectors
        empty = vectors - len(fd)
        for task, row in enumerate(fd):
            if not empty:
                width, count = heads, 1
            else:
                count = max(row['splits'] * heads // average, 1)
                width = (heads + count - 1) // count
                count = min((heads + width - 1) // width, empty + 1)
            for index in range(count):
                start = index * width
                length = width if index + 1 < count else heads - start
                fd_aiv.append([task, start, length])
                merge.append([row['sequence'], start, length])
            empty -= count - 1

    # The source traverses trailing zero-KV rows before the terminal endpoint.
    best[-1][0] = len(contexts) if capacity is None else capacity
    return {'contexts': list(contexts), 'heads': heads,
            'aic_owner_segments': assigned, 'aiv_merge_segments': merge,
            'native_host': {'used_aic': len(best), 'ends': best, 'fd': fd, 'fd_aiv': fd_aiv}}
