"""Two-level AICPU parallel executor's explicitly selected mesh-first lane.

[Role] Compose local Mesh RS, inter-group NHR RS/AG and local Mesh AG.
[Invariants] The source's route-0 split is 100%, with one equal-shard loop.
  Rank groups and remote notify events are explicit; local waits add no RTT.
  Internal NHR input/output/scratch aliases avoid invented intermediate copies.
[Timeouts & Locks] Finite O(R squared log R) contracts; no I/O or locks.
[Side-Effect Policy] No selector inference, device execution or timing fitting.
"""
from dataclasses import replace
from functools import lru_cache

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, HcclPlan


def _without(phases, names):
    """Remove true buffer aliases while preserving every dependency ancestor."""
    lookup = {p.name: p for p in phases}
    def ancestors(name):
        if name not in names:
            return (name,)
        return tuple(a for parent in lookup[name].after for a in ancestors(parent))
    return tuple(replace(p, after=tuple(dict.fromkeys(
        a for parent in p.after for a in ancestors(parent))))
        for p in phases if p.name not in names)


def _lift(matrix, groups, ranks):
    result = [[0] * ranks for _ in range(ranks)]
    for group in groups:
        for src, row in enumerate(matrix):
            for dst, value in enumerate(row):
                result[group[src]][group[dst]] = value
    return tuple(map(tuple, result))


def _notify(name, after, edges, rank, *, fused=False):
    sent = sum(src == rank for src, _ in edges)
    received = sum(dst == rank for _, dst in edges)
    if fused:
        # WRITE_REDUCE_WITH_NOTIFY already arrives with the final payload.
        return (HcclPhase(name + '_wait', after, (('hccl_control', received),),
            control_counts=(('remote_notify_fused_with_payload', sent),
                            ('remote_notify_wait', received))),)
    return (
        HcclPhase(name + '_record', after, (('hccl_control', sent),),
                  control_counts=(('remote_notify_record', sent),)),
        HcclPhase(name + '_signal', (name + '_record',), signals=edges),
        HcclPhase(name + '_wait', (name + '_signal',),
                  (('hccl_control', received),),
                  control_counts=(('remote_notify_wait', received),)),
    )


def _protocol(phases, rank, *, read=False, fused_reduce=False):
    """Pinned alg_data_trans_wrapper's ACK / payload / DATA_SIGNAL order."""
    result, tails = [], {}
    for phase in phases:
        deps = tuple(tails.get(p, p) for p in phase.after)
        if not phase.exchange:
            # These are intra-controller thread notifies, not network events.
            counters = tuple(('local_thread_' + k if k in {'notify', 'wait'} else k, v)
                             for k, v in phase.control_counts)
            result.append(replace(phase, after=deps, control_counts=counters))
            continue
        forward = tuple((src, dst) for src, row in enumerate(phase.exchange)
                        for dst, size in enumerate(row) if size)
        reverse = tuple((dst, src) for src, dst in forward)
        ack = forward if read else reverse
        done = reverse if read else forward
        prefix = _notify(phase.name + '_ack', deps, ack, rank)
        result.extend(prefix)
        result.append(replace(phase, after=(prefix[-1].name,)))
        suffix = _notify(phase.name + '_complete', (phase.name,), done, rank,
                         fused=fused_reduce)
        result.extend(suffix)
        tails[phase.name] = suffix[-1].name
    return tuple(result)


@lru_cache(maxsize=16)
def contract(selection, rank=0):
    from .hccl_aicpu import contract as mesh_contract
    from .hccl_nhr import contract as nhr_contract

    s = selection
    r, local, n = s.ranks, s.level0_size, s.payload_bytes
    if (s.expansion != 'AICPU_TS' or s.collective != 'allreduce'
            or s.template != 'parallel_mesh_nhr_route0' or s.topology_levels != 2
            or s.topology != 'MESH_1D' or s.buffer_mode != 'separate'
            or local is None or local < 2 or r % local or r // local < 2
            or not 0 <= rank < r):
        raise ModelingError('hccl_hierarchy_requires_explicit_two_level_mesh_route0')
    remote = r // local
    if remote & (remote - 1) or n % (r * s.dtype_bytes):
        raise ModelingError('hccl_hierarchy_requires_power_of_two_inter_equal_shards')
    if s.max_loop_bytes is None or n > s.max_loop_bytes:
        raise ModelingError('hccl_hierarchy_requires_declared_single_loop_capacity')
    if s.dtype_bytes == 8:
        raise ModelingError('hccl_hierarchy_64bit_reduce_not_implemented')
    if not n:
        return HcclPlan(s, rank, (HcclPhase('empty'),), 1, 0)
    intra = tuple(tuple(range(start, start + local)) for start in range(0, r, local))
    inter = tuple(tuple(range(offset, r, local)) for offset in range(local))
    base = replace(s, topology_levels=1, level0_size=None, max_loop_bytes=None,
                   signal_bytes=None)
    rs0 = mesh_contract(replace(base, collective='reduce_scatter', template='mesh',
                                ranks=local), rank % local)
    rs1 = nhr_contract(replace(base, collective='reduce_scatter', template='nhr',
                               ranks=remote, payload_bytes=n // local), rank // local)
    ag1 = nhr_contract(replace(base, collective='allgather', template='nhr',
                               ranks=remote, payload_bytes=n // r), rank // local)
    ag0 = mesh_contract(replace(base, collective='allgather', template='mesh',
                                ranks=local, payload_bytes=n // local), rank % local)
    # Executor RunTemplateInter0/Inter01 shares matching HCCL_BUFFER offsets;
    # no pre/post copy, and internal AG never enables its final read-to-output.
    rs1_phases = _without(rs1.phases, {'input_to_scratch', 'scratch_to_output',
                                     *(p.name for p in rs1.phases
                                       if p.name.endswith(('_ack', '_complete')))})
    ag1_phases = _without(ag1.phases, {'input_to_scratch', 'scratch_to_output',
                                     'output_copy_ready', *(p.name for p in ag1.phases
                                       if p.name.endswith(('_ack', '_complete')))})
    # Final Mesh AG input is already in scratch, but output is user memory.
    ag0_phases = tuple(replace(p, steps=p.steps[:1], work=p.steps[0][1])
                      if p.name == 'local_copies' else p for p in ag0.phases)
    specs = (('l0_rs', rs0.phases, intra, False, False),
             ('l1_rs', rs1_phases, inter, False, True),
             ('l1_ag', ag1_phases, inter, False, False),
             ('l0_ag', ag0_phases, intra, True, False))
    result, previous = [], ()
    for tag, phases, groups, read, fused in specs:
        # Two template-main threads remain allocated even when route1 is empty.
        # Their Pre/PostSync are local controller operations only.
        pre = tag + '_executor_pre'
        result.append(HcclPhase(pre, previous, (('hccl_control', 4),),
            control_counts=(('local_thread_notify', 2), ('local_thread_wait', 2))))
        lifted = tuple(replace(p, name=tag + '.' + p.name,
            after=tuple(tag + '.' + a for a in p.after) or (pre,),
            exchange=_lift(p.exchange, groups, r) if p.exchange else ()) for p in phases)
        expanded = _protocol(lifted, rank, read=read, fused_reduce=fused)
        result.extend(expanded)
        consumed = {a for p in expanded for a in p.after}
        tails = tuple(p.name for p in expanded if p.name not in consumed)
        post = tag + '_executor_post'
        result.append(HcclPhase(post, tails, (('hccl_control', 4),),
            control_counts=(('local_thread_notify', 2), ('local_thread_wait', 2))))
        previous = (post,)
    # Source reservation includes a conservative tail allowance even though
    # this selected schedule has equal shards (all_reduce_parallel_scratch.h).
    remainder = max(local, remote) - 1
    scratch = 2 * n + remainder * min(n // s.dtype_bytes, remainder) * s.dtype_bytes
    return HcclPlan(s, rank, tuple(result), 1 + max(local, local - 1) + 2,
        scratch, (('executor_local_sync_stages', 4), ('mesh_first_split_percent', 100)), (
            'explicit_single_loop_capacity_not_inferred_from_runtime_scratch',
            'route1_split_nonuniform_shards_multijetty_pcie_and_three_levels_not_implemented',
            'whole_phase_peer_barriers_not_per_thread_executor_overlap',
            'remote_signal_bytes_unspecified_propagation_only' if s.signal_bytes is None else
                'remote_signal_bytes_explicit_caller_declaration',
            'controller_issue_requires_declared_hccl_control_rate',
            'current_source_compatible_path_not_captured_executor_identity',
        ))
