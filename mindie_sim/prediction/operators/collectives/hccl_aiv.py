"""HCCL AIV Mesh1D all-reduce with finite producer and consumer ownership.

[Role] Price source copy/atomic paths independently of transport and selectors.
[Invariants] OneShot broadcasts N to R-1 peers; TwoShot exchanges shards twice.
  Atomic reduction is MTE3 read-modify-write, not vector arithmetic. Inactive
  AIVs cannot supply a busy owner. Phase barriers are an explicit approximation.
[Timeouts & Locks] O(ranks + AIV count) plans, finite UB copy windows; no I/O.
[Side-Effect Policy] Immutable ledgers and programs on the common Engine only.
"""
from dataclasses import asdict
import math

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HCCL_64BIT_ELEMENT_BYTES, HcclPhase, HcclPlan, HcclSelection, SOURCE
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage

RECIPE = 'hccl_aiv_source_phase'
GROUP_BUFFER = 'hccl_aiv_copy_queue'
UB_BATCH = 95 * 1024
UB_RESIDENCY = 190 * 1024 + 2560
FLAG_BYTES = 128
_ATOMIC_COPY_HBM_TRANSFERS = 3  # Source read, destination read and destination write.
_TWOSHOT_SCRATCH_PAYLOADS = 4


def _owner_demand(owners):
    demand = {}
    for copies in owners.values():
        for size, atomic in copies:
            for role, amount in _demand(size, atomic).items():
                demand[role] = demand.get(role, 0) + amount
    return tuple(demand.items())


def _flag_matrix(ranks):
    return _matrix((FLAG_BYTES,) * ranks) if ranks >= 2 else ()


def _flag_work(kind):
    return {'hbm': FLAG_BYTES, 'mte3' if kind == 'record' else 'mte2_aiv': FLAG_BYTES}


def _flag_owner_demand(owners):
    demand, counts = {}, {}
    for actions in owners.values():
        for kind in actions:
            for role, amount in _flag_work(kind).items():
                demand[role] = demand.get(role, 0) + amount
            counts[kind] = counts.get(kind, 0) + 1
    return tuple(demand.items()), tuple(sorted(counts.items()))


def _oneshot_flag_owners(ranks, rank):
    remote = tuple(peer for peer in range(ranks) if peer != rank)
    if not remote:
        return {}
    return {
        'pre_sync': {peer: ('record',) for peer in remote},
        'pre_sync_wait': {peer: ('wait',) for peer in remote},
        'data_ready': {peer: ('record',) for peer in remote},
        'data_ready_wait': {ranks: ('wait',) * len(remote)},
        'post_barrier': {peer: ('record',) for peer in remote},
        'post_barrier_wait': {peer: ('wait',) for peer in remote},
    }


def _flag_phase(name, after, owners, *, exchange=()):
    work, counts = _flag_owner_demand(owners)
    return HcclPhase(
        name, after, work, exchange=exchange,
        owner_domain=tuple(owner for owner, actions in owners.items() if actions),
        control_counts=counts,
    )


def _flag_stages(flag_selected, context, units):
    stages = []
    for owner, actions in flag_selected.items():
        if not actions:
            continue
        inner = []
        for idx, kind in enumerate(actions):
            demand = Work(resources=_flag_work(kind))
            inner.append(Stage(
                f'{kind}_{idx}', context.bind(demand),
                after=(f'{actions[idx - 1]}_{idx - 1}',) if idx else (),
                placement=context.owned(owner, units, demand.resources),
            ))
        allocations = (() if GROUP_BUFFER in context.group_buffers else (
            context.allocate(f'owner{owner}:copy', 'ub', UB_RESIDENCY, (owner,)),))
        stages.append(Stage(
            f'owner{owner}', program=KernelProgram(tuple(inner)),
            acquire=allocations, free=allocations,
        ))
    return tuple(stages)


def blocks(selection, available):
    limit = selection.block_limit or available
    if limit > available:
        raise ModelingError('hccl_aiv_block_limit_exceeds_available_owners')
    r = selection.ranks
    count = min(limit, r + 1) if selection.template == 'mesh_oneshot' else limit // (r + 1) * (r + 1)
    if count < (r + 1 if selection.template == 'mesh_oneshot' else 2 * r):
        raise ModelingError('hccl_aiv_small_core_branch_not_selected')
    return count


def _demand(size, atomic=False):
    return {'hbm': size * (_ATOMIC_COPY_HBM_TRANSFERS if atomic else 2),
            'mte2_aiv': size, 'mte3': size}


def _matrix(values, destination=False):
    return tuple(tuple(0 if i == j else values[j if destination else i]
                       for j in range(len(values))) for i in range(len(values)))


def jobs(selection, available, rank=0):
    """Each owner executes ordered (bytes, atomic) GM-to-GM copies."""
    s = selection
    if (s.expansion != 'AIV' or s.collective != 'allreduce' or s.topology != 'MESH_1D'
            or s.topology_levels not in {1, 2} or s.template not in {'mesh_oneshot','mesh_twoshot'}
            or not 0 <= rank < s.ranks):
        raise ModelingError('hccl_aiv_requires_explicit_mesh_allreduce')
    if s.dtype_bytes == HCCL_64BIT_ELEMENT_BYTES:
        raise ModelingError('hccl_aiv_int64_reduce64_program_not_selected')
    if s.buffer_mode != 'separate':
        raise ModelingError('hccl_aiv_requires_source_separate_buffers')
    r, n = s.ranks, s.payload_bytes
    count = blocks(s, available)
    if s.template == 'mesh_oneshot':
        return {'broadcast': {i: ((n, False),) for i in range(r)},
                'reduce_output': {r: ((n, False),) + ((n, True),) * (r - 1)}}, count, (n,) * r
    group = (count - r) // r
    small, tail = divmod(n // s.dtype_bytes, group * r)
    sizes = tuple((small + (i < tail)) * s.dtype_bytes for i in range(group * r))
    chunks = tuple(sum(sizes[i*group:(i+1)*group]) for i in range(r))
    result = {
        'precopy': {i: tuple((size, False) for size in sizes[i*group:(i+1)*group]) for i in range(r)},
        'get_remote': {r + i*group + j: ((sizes[rank*group+j], False),)
                       for i in range(r) for j in range(group)},
        'reduce': {r + rank*group+j: ((sizes[rank*group+j], True),) * (r-1) for j in range(group)},
        'gather_output': {r + i*group+j: ((sizes[i*group+j], False),)
                          for i in range(r) for j in range(group)},
    }
    return result, count, chunks


def _twoshot_lane_slices(selection, count):
    r, n = selection.ranks, selection.payload_bytes
    group = (count - r) // r
    small, tail = divmod(n // selection.dtype_bytes, group * r)
    sizes = tuple((small + (i < tail)) * selection.dtype_bytes for i in range(group * r))
    return group, sizes


def contract(selection, available, rank=0):
    if isinstance(available, bool) or not isinstance(available, int) or available <= 0:
        raise ModelingError('hccl_aiv_requires_positive_available_owners')
    owner_jobs, count, chunks = jobs(selection, available, rank)
    r = selection.ranks
    unmodeled = (
        'TopoMatchOneLevel_full_group_peer_access_declared; physical_levels_do_not_add_algorithm_rounds',
        'extra_flag_poll_retries_not_timed',
        'whole_phase_exchange_barriers_replace_per_slice_peer_readiness',
        'executor_scratch_segmentation_and_first_operator_barrier_not_selected',
    )
    if not selection.payload_bytes:
        return HcclPlan(selection, rank, (HcclPhase('empty'),), count, 0, unmodeled=unmodeled)
    if selection.template == 'mesh_oneshot':
        flag_ex = _flag_matrix(r)
        flag_owners = _oneshot_flag_owners(r, rank)
        broadcast_ex = _matrix(chunks) if r >= 2 and any(chunks) else ()
        phases = [
            _flag_phase('pre_sync', (), flag_owners.get('pre_sync', {}), exchange=flag_ex),
            _flag_phase('pre_sync_wait', ('pre_sync',), flag_owners.get('pre_sync_wait', {})),
            HcclPhase('broadcast', ('pre_sync_wait',), _owner_demand(owner_jobs['broadcast']),
                      exchange=broadcast_ex, owner_domain=tuple(owner_jobs['broadcast'])),
            _flag_phase('data_ready', ('broadcast',), flag_owners.get('data_ready', {}), exchange=flag_ex),
            _flag_phase('data_ready_wait', ('data_ready',), flag_owners.get('data_ready_wait', {})),
            HcclPhase('reduce_output', ('data_ready_wait',), _owner_demand(owner_jobs['reduce_output']),
                      owner_domain=tuple(owner_jobs['reduce_output'])),
            _flag_phase('post_barrier', ('reduce_output',), flag_owners.get('post_barrier', {}), exchange=flag_ex),
            _flag_phase('post_barrier_wait', ('post_barrier',), flag_owners.get('post_barrier_wait', {})),
        ]
    else:
        group, sizes = _twoshot_lane_slices(selection, count)
        phases = [
            HcclPhase('precopy', (), _owner_demand(owner_jobs['precopy']),
                      owner_domain=tuple(owner_jobs['precopy'])),
            HcclPhase('get_remote', ('precopy',), _owner_demand(owner_jobs['get_remote']),
                      owner_domain=tuple(owner_jobs['get_remote'])),
        ]
        get_lanes = []
        for lane in range(group):
            lane_slices = tuple(sizes[i * group + lane] for i in range(r))
            if r >= 2 and any(lane_slices):
                lane_name = f'get_remote_lane_{lane}'
                phases.append(HcclPhase(
                    lane_name, ('precopy',), (),
                    exchange=_matrix(lane_slices, destination=True),
                    native_direction='read',
                ))
                get_lanes.append(lane_name)
        phases.append(HcclPhase(
            'reduce', ('get_remote', *get_lanes), _owner_demand(owner_jobs['reduce']),
            owner_domain=tuple(owner_jobs['reduce']),
        ))
        phases.append(HcclPhase(
            'gather_output', ('reduce',), _owner_demand(owner_jobs['gather_output']),
            owner_domain=tuple(owner_jobs['gather_output']),
        ))
        for lane in range(group):
            lane_slices = tuple(sizes[i * group + lane] for i in range(r))
            if r >= 2 and any(lane_slices):
                phases.append(HcclPhase(
                    f'gather_output_lane_{lane}', ('reduce',), (),
                    exchange=_matrix(lane_slices),
                    native_direction='read',
                ))
    return HcclPlan(selection, rank, tuple(phases), count,
                    selection.payload_bytes * (
                        selection.ranks if selection.template == 'mesh_oneshot'
                        else _TWOSHOT_SCRATCH_PAYLOADS),
                    unmodeled=unmodeled)


def build(op, geometry):
    raw = dict(op.metadata)['hccl_phase']
    s = HcclSelection(**raw['selection'])
    units = geometry.resource_units.get('vector', geometry.compute_units)
    rank = raw.get('rank', 0)
    owner_jobs, count, _ = jobs(s, units, rank)
    plan = contract(s, units, rank)
    phase = next(p for p in plan.phases if p.name == raw['phase'])
    flag_selected = (
        _oneshot_flag_owners(s.ranks, rank).get(phase.name, {})
        if s.template == 'mesh_oneshot' and s.payload_bytes else {}
    )
    selected = owner_jobs.get(phase.name, {})
    work = Work(resources=dict(phase.work))
    capacity = geometry.buffers.get('ub',0)
    if capacity < UB_RESIDENCY:
        raise ModelingError('hccl_aiv_requires_full_copy_queue_and_flag_buffer')

    def program(context):
        if flag_selected:
            return KernelProgram(_flag_stages(flag_selected, context, units))
        owners = []
        for owner, copies in selected.items():
            # Every CpGM2GM call iterates read then write through a finite UB
            # queue; its misleading useDoubleBuffer flag halves the batch size,
            # but this source uses a single input queue slot.
            sizes = tuple((size, atomic) for size, atomic in copies if size)
            prefix = [0]
            for size, _ in sizes:
                prefix.append(prefix[-1] + math.ceil(size / UB_BATCH))

            def copy_stages(amount, atomic):
                read = Work(resources={'hbm': amount,'mte2_aiv': amount})
                write = Work(resources={'hbm': amount*(2 if atomic else 1),'mte3': amount})
                return (
                    Stage('read',context.bind(read),placement=context.owned(owner,units,read.resources)),
                    Stage('write',context.bind(write),after=('read',),
                          placement=context.owned(owner,units,write.resources)),
                )

            # Repeated full UB batches have identical bound work and owner
            # placement. Reuse those stages and each source call's real tail;
            # the one-slot pipeline still issues every read before its write.
            shapes = {(min(size, UB_BATCH), atomic) for size, atomic in sizes}
            shapes.update((size % UB_BATCH, atomic) for size, atomic in sizes
                          if size % UB_BATCH)
            templates = {shape: copy_stages(*shape) for shape in shapes}

            def tile(index, *, sizes=sizes, prefix=tuple(prefix), templates=templates):
                import bisect
                call = bisect.bisect_right(prefix, index)-1
                size, atomic = sizes[call]
                amount = min(UB_BATCH, size-(index-prefix[call])*UB_BATCH)
                return templates[amount, atomic]
            if prefix[-1]:
                allocations = (() if GROUP_BUFFER in context.group_buffers else (
                    context.allocate(f'owner{owner}:copy', 'ub', UB_RESIDENCY, (owner,)),))
                owners.append(Stage(f'owner{owner}',
                    program=PipelineProgram(prefix[-1],1,tile),
                    acquire=allocations, free=allocations))
        return KernelProgram(tuple(owners))
    return OperatorPlan(work,program,{
        'source':SOURCE,'hccl_selection':asdict(s),'hccl_phase':phase.name,
        'execution_class':'vector','launched_blocks':count,
        'active_owner_domain':phase.owner_domain,'scratch_bytes':plan.scratch_bytes,
        'buffer_model':'active_copy_queue_and_flags; unused_Reduce64_queues_not_priced',
        'unmodeled_services':plan.unmodeled,
        'coverage':'source_copy_atomic_phase_with_explicit_block_limit',
    })
