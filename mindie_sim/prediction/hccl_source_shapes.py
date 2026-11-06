"""Shared exact source-shape arithmetic for HCCL AICPU mesh templates.

[Role] Build phase-level work and exchange records from the pinned AR/RS/AG
  source formulas without rates, measurements or a second executor.
[Invariants] Byte amounts are positive integers. AR shards/chunks are
  element-exact; RS mesh-chunk chunks use 4 KiB floor plus exact tail. Ordered
  segment/channel sequences are preserved. Exchange matrices are physical
  [source][destination] sends; native direction is metadata only.
[Timeouts & Locks] O(ranks squared * chunks) finite metadata; no locks.
[Side-Effect Policy] Pure constructors; no I/O, device or global state.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, exact_positive_int

ALIGN_BYTES = 4096


def _work_sum(steps):
    values = {}
    for _, work in steps:
        for role, amount in work:
            values[role] = values.get(role, 0) + amount
    return tuple(sorted(values.items()))


def _nonnegative_int(value, label):
    import math
    if isinstance(value, bool):
        raise ModelingError(f"{label}_requires_nonnegative_integer:{value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ModelingError(f"{label}_requires_nonnegative_integer:{value!r}")
    if number < 0:
        raise ModelingError(f"{label}_requires_nonnegative_integer:{value!r}")
    return number


def copy_work(size):
    return (("hbm", 2 * size), ("hccl_copy", size)) if size else ()


def reduce_work(size):
    return (("hbm", 3 * size), ("hccl_reduce", size)) if size else ()


def copy_phase(name, after, size):
    return HcclPhase(name, after, copy_work(size), native_direction="local")


def reduce_phase(name, after, size, steps):
    steps = tuple((f"{name}_{index}", reduce_work(size)) for index in range(steps))
    return HcclPhase(name, after, _work_sum(steps), steps=steps,
                     native_direction="local")


def sync_phase(name, after, threads):
    notifies = max(0, threads - 1)
    return HcclPhase(name, after, (("hccl_control", 2 * notifies),) if notifies else (),
                     control_counts=(("notify", notifies), ("wait", notifies)),
                     native_direction="local")


def exchange_phase(name, after, matrix, rank, *, remote_reduce=False,
                   native_direction="send"):
    r = len(matrix)
    if any(len(row) != r for row in matrix):
        raise ModelingError("hccl_exchange_matrix_not_square")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ModelingError("hccl_exchange_requires_nonnegative_integer_bytes")
    if any(matrix[index][index] for index in range(r)):
        raise ModelingError("hccl_exchange_diagonal_must_be_empty")
    sent = sum(matrix[rank])
    received = sum(row[rank] for row in matrix)
    work = {"hbm": sent + (2 * received if remote_reduce else received)}
    if remote_reduce:
        work["hccl_reduce"] = received
    outgoing = sum(value > 0 for value in matrix[rank])
    return HcclPhase(
        name, after, tuple(sorted(work.items())),
        exchange=tuple(tuple(row) for row in matrix),
        control_counts=(("directed_peer_messages", outgoing),
                        ("payload_slices", outgoing)),
        native_direction=native_direction)


def slice_sizes(payload_bytes, dtype_bytes, ranks):
    """AR CalcSliceInfoVec: ceil(count/R) elements; final rank takes the tail."""
    payload_bytes = exact_positive_int(payload_bytes, "hccl_payload_bytes")
    dtype_bytes = exact_positive_int(dtype_bytes, "hccl_dtype_bytes")
    ranks = exact_positive_int(ranks, "hccl_ranks")
    count = payload_bytes // dtype_bytes
    chunk = (count + ranks - 1) // ranks
    sizes, accumulated = [], 0
    for _ in range(ranks):
        take = min(chunk, count - accumulated)
        sizes.append(take * dtype_bytes)
        accumulated += take
    if accumulated != count:
        raise ModelingError("hccl_slice_conservation_failed")
    return tuple(sizes)


def subchunks(shard_bytes, dtype_bytes, ranks):
    """AR source 182-198: R-1 element-exact chunks, big chunks first."""
    shard_bytes = _nonnegative_int(shard_bytes, "hccl_shard_bytes")
    dtype_bytes = exact_positive_int(dtype_bytes, "hccl_dtype_bytes")
    ranks = exact_positive_int(ranks, "hccl_ranks")
    if ranks < 2:
        raise ModelingError("hccl_subchunks_requires_two_ranks")
    count = shard_bytes // dtype_bytes
    slice_num = ranks - 1
    big_num = count % slice_num
    big_size = (count // slice_num + 1) * dtype_bytes
    small_size = (count // slice_num) * dtype_bytes
    return tuple([big_size] * big_num + [small_size] * (slice_num - big_num))


def rs_chunks(shard_bytes, dtype_bytes, ranks):
    """RS mesh-chunk source 181-207: 4 KiB floor plus exact remaining tail."""
    shard_bytes = exact_positive_int(shard_bytes, "hccl_shard_bytes")
    dtype_bytes = exact_positive_int(dtype_bytes, "hccl_dtype_bytes")
    ranks = exact_positive_int(ranks, "hccl_ranks")
    if ranks < 2:
        raise ModelingError("hccl_rs_chunks_requires_two_ranks")
    count = shard_bytes // dtype_bytes
    slice_num = ranks - 1
    aligned = (shard_bytes // slice_num) // ALIGN_BYTES * ALIGN_BYTES
    if slice_num >= 2 and aligned > 0:
        return tuple([aligned] * (slice_num - 1)
                     + [shard_bytes - aligned * (slice_num - 1)])
    big_num = count % slice_num
    big_size = (count // slice_num + 1) * dtype_bytes
    small_size = (count // slice_num) * dtype_bytes
    return tuple([big_size] * big_num + [small_size] * (slice_num - big_num))


def step_matrix(chunks, step, ranks):
    """AR source 207-245: one chunk per peer, rotated peer map."""
    matrix = [[0] * ranks for _ in range(ranks)]
    for source in range(ranks):
        for chunk_index in range(ranks - 1):
            next_num = step + chunk_index + 1
            if next_num >= ranks:
                next_num += 1
            next_rank = (source + next_num) % ranks
            pre_rank = (2 * source + ranks - next_rank) % ranks
            if matrix[source][pre_rank]:
                raise ModelingError("hccl_mesh_chunk_peer_slot_duplicated")
            matrix[source][pre_rank] = chunks[pre_rank][chunk_index]
    return tuple(tuple(row) for row in matrix)


def rs_step_matrix(chunks, step, ranks):
    """RS mesh-chunk source 226-239: send chunk i to frontRank."""
    matrix = [[0] * ranks for _ in range(ranks)]
    for source in range(ranks):
        for chunk_index in range(ranks - 1):
            next_num = step + chunk_index + 1
            if next_num >= ranks:
                next_num += 1
            next_rank = (source + next_num) % ranks
            front_rank = (2 * source - next_rank + ranks) % ranks
            if matrix[source][front_rank]:
                raise ModelingError("hccl_rs_mesh_chunk_peer_slot_duplicated")
            matrix[source][front_rank] = chunks[chunk_index]
    return tuple(tuple(row) for row in matrix)


def detour_split(total_bytes, dtype_bytes, level0_weight, level1_weights):
    """Z detour template_utils 146-220, exact integer/4 KiB rules."""
    total_bytes = exact_positive_int(total_bytes, "hccl_detour_total_bytes")
    dtype_bytes = exact_positive_int(dtype_bytes, "hccl_dtype_bytes")
    level0_weight = exact_positive_int(level0_weight, "hccl_level0_weight")
    level1_weights = tuple(exact_positive_int(weight, "hccl_level1_weight")
                           for weight in level1_weights)
    if total_bytes % dtype_bytes:
        raise ModelingError("hccl_detour_requires_integer_bytes")
    if not level1_weights:
        return (total_bytes,)
    count = total_bytes // dtype_bytes
    total_weight = level0_weight + sum(level1_weights)

    def split(portion, weight, weight_total):
        value = (portion // weight_total * weight
                 + portion % weight_total * weight // weight_total)
        align = ALIGN_BYTES // dtype_bytes
        aligned = value // align * align
        return aligned if aligned else value

    level0 = split(count, level0_weight, total_weight)
    remaining = count - level0
    result = [level0]
    level1_total = count - level0
    for index, weight in enumerate(level1_weights):
        value = (remaining if index == len(level1_weights) - 1
                 else split(level1_total, weight, sum(level1_weights)))
        result.append(value)
        remaining -= value
    if any(value < 0 for value in result):
        raise ModelingError("hccl_detour_split_negative")
    out = tuple(value * dtype_bytes for value in result)
    if sum(out) != total_bytes:
        raise ModelingError("hccl_detour_split_conservation_failed")
    return out
