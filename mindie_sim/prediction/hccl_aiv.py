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
from .hccl_contracts import HcclPhase, HcclPlan, HcclSelection, SOURCE
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, PipelineProgram, Stage

RECIPE = 'hccl_aiv_source_phase'
GROUP_BUFFER = 'hccl_aiv_copy_queue'
UB_BATCH = 95 * 1024
UB_RESIDENCY = 190 * 1024 + 2560


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
    return {'hbm': size * (3 if atomic else 2), 'mte2_aiv': size, 'mte3': size}


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
    if s.dtype_bytes == 8:
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


def contract(selection, available=64, rank=0):
    owner_jobs, count, chunks = jobs(selection, available, rank)
    phases, previous = [], ()
    for name, owners in owner_jobs.items():
        demand = {}
        for copies in owners.values():
            for size, atomic in copies:
                for role, amount in _demand(size, atomic).items():
                    demand[role] = demand.get(role, 0) + amount
        exchange = (_matrix(chunks) if name in {'broadcast','gather_output'} else
                    _matrix(chunks, destination=True) if name == 'get_remote' else ())
        phases.append(HcclPhase(name, previous, tuple(demand.items()), exchange=exchange,
                               owner_domain=tuple(owners)))
        previous = (name,)
    return HcclPlan(selection, rank, tuple(phases), count,
                    selection.payload_bytes * (selection.ranks if selection.template == 'mesh_oneshot' else 4),
                    unmodeled=(
                        'TopoMatchOneLevel_full_group_peer_access_declared; physical_levels_do_not_add_algorithm_rounds',
                        'successful_flag_operations_and_poll_retries_not_timed',
                        'whole_phase_exchange_barriers_replace_per_slice_peer_readiness',
                        'executor_scratch_segmentation_and_first_operator_barrier_not_selected',
                    ))


def build(op, geometry):
    raw = dict(op.metadata)['hccl_phase']
    s = HcclSelection(**raw['selection'])
    units = geometry.resource_units.get('vector', geometry.compute_units)
    owner_jobs, count, _ = jobs(s, units, raw.get('rank',0))
    plan = contract(s, units, raw.get('rank',0))
    phase = next(p for p in plan.phases if p.name == raw['phase'])
    selected = owner_jobs[phase.name]
    work = Work(resources=dict(phase.work))
    capacity = geometry.buffers.get('ub',0)
    if capacity < UB_RESIDENCY:
        raise ModelingError('hccl_aiv_requires_full_copy_queue_and_flag_buffer')

    def program(context):
        owners = []
        for owner, copies in selected.items():
            # Every CpGM2GM call iterates read then write through a finite UB
            # queue; its misleading useDoubleBuffer flag halves the batch size,
            # but this source uses a single input queue slot.
            sizes = tuple((size, atomic) for size, atomic in copies if size)
            prefix = [0]
            for size, _ in sizes:
                prefix.append(prefix[-1] + math.ceil(size / UB_BATCH))

            def tile(index, *, owner=owner, sizes=sizes, prefix=tuple(prefix)):
                import bisect
                call = bisect.bisect_right(prefix, index)-1
                size, atomic = sizes[call]
                amount = min(UB_BATCH, size-(index-prefix[call])*UB_BATCH)
                read = Work(resources={'hbm': amount,'mte2_aiv': amount})
                write = Work(resources={'hbm': amount*(2 if atomic else 1),'mte3': amount})
                return (
                    Stage('read',context.bind(read),placement=context.owned(owner,units,read.resources)),
                    Stage('write',context.bind(write),after=('read',),
                          placement=context.owned(owner,units,write.resources)),
                )
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
