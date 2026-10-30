"""Pinned AICPU NHR rounds with source buffer optimizations.

[Role] Express single-channel non-PCIe AR/AG/RS on the common HCCL phase DAG.
[Invariants] Cyclic source peers are not XOR partners. Equal integer shards and
  power-of-two ranks retain exact directed traffic and per-round dependencies.
  RS skips first-round transmit PreCopy; AG reads its last round into output.
  Remote reduce counts sender read plus destination read/write, never two wires.
[Timeouts & Locks] O(R squared log R) finite metadata; no locks or external I/O.
[Side-Effect Policy] Pure work and dependency contracts; no measured constants.
"""
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, HcclPlan


@dataclass(frozen=True)
class NhrRound:
    send_to: int
    receive_from: int
    send_slices: tuple[int, ...]
    receive_slices: tuple[int, ...]


def rounds(ranks, rank, kind):
    """AR:640–729, AG:422–453, RS:447–499 at pinned c2d67904.

    This implemented domain uses no virtual ranks or rounded fractional shards.
    The general non-power-of-two source schedule remains outside this contract.
    """
    count = (ranks - 1).bit_length()
    result = []
    for index in range(count):
        bit = index if kind == 'reduce_scatter' else count - 1 - index
        distance = 1 << bit
        width = ranks // (2 * distance)
        send = (rank - distance if kind == 'reduce_scatter' else rank + distance) % ranks
        receive = (rank + distance if kind == 'reduce_scatter' else rank - distance) % ranks
        tx = send if kind == 'reduce_scatter' else rank
        rx = rank if kind == 'reduce_scatter' else receive
        result.append(NhrRound(send, receive,
            tuple((tx - i * 2 * distance) % ranks for i in range(width)),
            tuple((rx - i * 2 * distance) % ranks for i in range(width))))
    return tuple(result)


def _copy(name, after, size):
    return HcclPhase(name, after, (('hbm', 2 * size), ('hccl_copy', size)))


def _sync(name, after):
    # One notify and its wait. Without a declared control rate, these remain
    # readiness plus explicit unpriced work, not fitted fixed microseconds.
    return HcclPhase(name, after, (('hccl_control', 2),),
                     control_counts=(('notify', 1), ('wait', 1)))


def contract(selection, rank=0):
    s = selection
    r, n = s.ranks, s.payload_bytes
    if (s.expansion != 'AICPU_TS' or s.template != 'nhr'
            or s.topology not in {'MESH_1D', 'MESH_1D_CLOS'}
            or s.topology_levels != 1 or not 0 <= rank < r):
        raise ModelingError('hccl_nhr_requires_single_level_aicpu_communicator')
    if r & (r - 1):
        raise ModelingError('hccl_nhr_non_power_of_two_schedule_not_implemented')
    if s.collective not in {'allreduce', 'allgather', 'reduce_scatter'}:
        raise ModelingError('hccl_nhr_collective_not_implemented')
    if s.buffer_mode != 'separate':
        raise ModelingError('hccl_nhr_requires_distinct_input_scratch_output')
    if s.dtype_bytes == 8 and s.collective != 'allgather':
        raise ModelingError('hccl_nhr_64bit_reduce_path_not_implemented')
    if s.collective != 'allgather' and n % (r * s.dtype_bytes):
        raise ModelingError('hccl_nhr_tail_shards_not_implemented')
    if not n:
        return HcclPlan(s, rank, (HcclPhase('empty'),), 1, 0)
    if r == 1:
        phases = (_copy('input_to_output', (), n),) if s.collective == 'allreduce' else (
            _copy('input_to_scratch', (), n), _copy('scratch_to_output', ('input_to_scratch',), n))
        return HcclPlan(s, rank, phases, 1 if s.collective != 'allgather' else 2, n)

    shard = n if s.collective == 'allgather' else n // r
    total = r * shard
    phases = []
    previous = ()

    def run(kind, *, parallel_final_copy=False):
        nonlocal previous
        schedules = tuple(rounds(r, source, kind) for source in range(r))
        for index, own in enumerate(schedules[rank]):
            tag = ('rs' if kind == 'reduce_scatter' else 'ag') + f'_{index}'
            last = index == len(schedules[rank]) - 1
            if last and parallel_final_copy:
                # AG's second thread copies already resident shards while the
                # communication thread reads final remote shards into output.
                phases.append(_sync('output_copy_ready', previous))
                previous = ('output_copy_ready',)
                phases.append(_copy('scratch_to_output', previous,
                                    total - len(own.receive_slices) * shard))
            phases.append(_sync(tag + '_ack', previous))
            matrix = tuple(tuple(len(schedules[src][index].send_slices) * shard
                                 if dst == schedules[src][index].send_to else 0
                                 for dst in range(r)) for src in range(r))
            sent, received = sum(matrix[rank]), sum(row[rank] for row in matrix)
            reduce = kind == 'reduce_scatter'
            work = (('hbm', sent + received * (2 if reduce else 1)),)
            if reduce:
                work += (('hccl_reduce', received),)
            phases.append(HcclPhase(tag + '_exchange', (tag + '_ack',), work,
                exchange=matrix,
                control_counts=(('directed_peer_messages', 1),
                                ('payload_slices', len(own.send_slices)))))
            phases.append(_sync(tag + '_complete', (tag + '_exchange',)))
            previous = (tag + '_complete',)

    if s.collective == 'allgather':
        phases.append(_sync('thread_pre_sync', ()))
        phases.append(_copy('input_to_scratch', ('thread_pre_sync',), n))
        previous = ('input_to_scratch',)
        run('allgather', parallel_final_copy=True)
        phases.append(_sync('thread_post_sync', (*previous, 'scratch_to_output')))
    else:
        # First RS write-reduce reads outgoing shards straight from user input.
        first_tx = rounds(r, rank, 'reduce_scatter')[0].send_slices
        phases.append(_copy('input_to_scratch', (), total - len(first_tx) * shard))
        previous = ('input_to_scratch',)
        run('reduce_scatter')
        if s.collective == 'allreduce':
            run('allgather')
            last_rx = rounds(r, rank, 'allgather')[-1].receive_slices
            phases.append(_copy('scratch_to_output', previous, total - len(last_rx) * shard))
        else:
            phases.append(_copy('scratch_to_output', previous, shard))
    return HcclPlan(s, rank, tuple(phases), 2 if s.collective == 'allgather' else 1, total,
        unmodeled=(
            'pcie_read_modes_symmetric_buffers_and_multiple_channels_not_implemented',
            'capture_executor_dispatch_and_binary_identity_not_established',
            'non_power_of_two_ranks_tail_shards_and_multilevel_execution_not_implemented',
            'ack_data_signal_wire_bytes_and_task_issue_without_physical_declaration',
            'source_cost_model_empirical_microsecond_coefficients_not_used',
            'scratch_capacity_and_executor_message_segmentation_not_scheduled',
        ))
