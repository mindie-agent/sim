"""Pinned AIV Mesh1D AllGather producer and consumer copy programs.

[Role] Preserve AllGather's own block partition, successful flags and copy slots.
[Invariants] CalNumBlocks keeps the requested grid. Small messages use at most
  R copy owners; large messages use R producers and a separate consumer grid.
  Input is copied into CCL once, and R CCL inputs are read into output once.
[Timeouts & Locks] O(R * active blocks) actions; payload tiles issue lazily.
[Side-Effect Policy] Common Engine only; no device I/O or measured timing fits.
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass
from math import ceil
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from .hccl_aiv import GROUP_BUFFER, UB_BATCH, UB_RESIDENCY
from .hccl_contracts import HcclPhase, HcclPlan, HcclSelection, SOURCE, SOURCE_REVISION
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/all_gather/algorithm/template/aiv/kernel/aiv_all_gather_mesh_1d.h",
        source_sha256="3ca3049ddf8bacaa688e030f71641044593d5657b2644ea8167f31a966a5bb25",
    ),
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/all_gather/algorithm/template/aiv/aiv_temp_all_gather_mesh_1D.cc",
        source_sha256="abe686271bb67ab44c7a830d6d7fa00bae56a8afe1ca92d83fc4819ba7c1c61a",
    ),
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/op_common/algorithm/template/aiv/aiv_communication_base_v2.h",
        source_sha256="09fa4eabd485d8f7f24cc3a1225a91cf6373734f7bf38f3fa35db4cd5d156fa2",
    ),
)

RECIPE = 'hccl_aiv_gather_source_phase'
DATA_LIMIT = 512 * 1024
PINGPONG_TOTAL_LIMIT = 8 * 1024 * 1024
FLAG_BYTES = 128


@dataclass(frozen=True)
class Action:
    kind: str
    size: int

    def work(self):
        if self.kind == 'copy':
            return {'hbm': 2*self.size, 'mte2_aiv': self.size, 'mte3': self.size}
        return {'hbm': self.size,
                'mte3' if self.kind == 'record' else 'mte2_aiv': self.size}


def _partition(n, count):
    q, remainder = divmod(n, count)
    return tuple(q+(i < remainder) for i in range(count))


def jobs(s, available, rank=0):
    """Ordered owner actions for a single normal-kernel executor slice."""
    if isinstance(available, bool) or not isinstance(available, int) or available <= 0:
        raise ModelingError('hccl_aiv_gather_requires_positive_available_owners')
    if (s.expansion != 'AIV' or s.collective != 'allgather' or s.template != 'mesh'
            or s.topology != 'MESH_1D' or s.topology_levels not in {1, 2}
            or s.source_revision != SOURCE_REVISION or s.buffer_mode != 'separate'
            or s.ranks < 2 or not 0 <= rank < s.ranks):
        raise ModelingError('hccl_aiv_gather_requires_explicit_separate_Mesh1D_normal_kernel')
    count = s.block_limit or available
    if not 1 <= count <= available:
        raise ModelingError('hccl_aiv_gather_block_limit_exceeds_available_owners')
    r, n, width = s.ranks, s.payload_bytes, s.dtype_bytes
    if n % width:
        raise ModelingError('hccl_aiv_gather_requires_integer_elements')
    big = n >= DATA_LIMIT and count >= 2*r
    pingpong = n <= DATA_LIMIT and r*n <= PINGPONG_TOTAL_LIMIT
    result = {'precopy': {}, 'gather': {}}
    pieces = 0
    if big:
        usable = count//r*r
        cut = (usable-r)//r
        sizes = tuple(_partition(part, cut) for part in _partition(n//width, r))
        pieces = sum(size > 0 for group in sizes for size in group)
        for owner, group in enumerate(sizes):
            result['precopy'][owner] = tuple(action for size in group if size
                for action in (Action('copy',size*width),Action('record',FLAG_BYTES)))
        for peer in range(r):
            for lane in range(cut):
                result['gather'][r+peer*cut+lane] = tuple(
                    action for idx in range(r) if (size := sizes[(idx+rank)%r][lane])
                    for action in (Action('wait',FLAG_BYTES),Action('copy',size*width)))
        copy_owners = usable
    else:
        copy_owners = min(count,r)
        q = (n//width)//copy_owners
        # RunCtrlCore places the whole remainder on its last producer.
        for owner in range(copy_owners):
            size = ((n//width)-q*(copy_owners-1) if owner == copy_owners-1 else q)*width
            result['precopy'][owner] = (Action('copy',size),) if size else ()
            peers = _partition(r,copy_owners)[owner]
            result['gather'][owner] = (
                (Action('record',FLAG_BYTES),)*peers
                + tuple(action for _ in range(peers) for action in (
                    Action('wait',FLAG_BYTES),Action('copy',n),Action('record',FLAG_BYTES)))
                + (Action('wait',FLAG_BYTES),)*peers)
    if not pingpong:
        # BarrierAll's SyncAll precedes these flags. Every launched block joins
        # SyncAll; only min(R, count) blocks own this rank-partitioned traffic.
        result['post_barrier'] = {}
        for owner, peers in enumerate(_partition(r,count)):
            if peers:
                result['post_barrier'][owner] = ((Action('record',FLAG_BYTES),)*peers
                    + tuple(a for _ in range(peers) for a in (
                        Action('wait',FLAG_BYTES),Action('record',FLAG_BYTES))))
    return result, count, {
        'branch': 'producer_consumer' if big else 'control_core',
        'copy_owner_count': copy_owners, 'pingpong': pingpong,
        'positive_input_fragments': pieces, 'input_bytes': n,
        'sync_all_calls_per_launched_block': int(not big)+int(not pingpong),
    }


def _mesh(ranks, wire):
    if not wire:
        return ()
    return tuple(tuple(0 if i == j else wire for j in range(ranks)) for i in range(ranks))


def _producer_fragment_bytes(selection, count):
    r, n, width = selection.ranks, selection.payload_bytes, selection.dtype_bytes
    usable = count // r * r
    cut = (usable - r) // r
    sizes = tuple(_partition(part, cut) for part in _partition(n // width, r))
    return tuple(size * width for group in sizes for size in group if size)


def _phase_specs(owner_jobs, selection, count, details):
    """Emit causally ordered flag and payload phases without coalescing transfers."""
    r, n = selection.ranks, selection.payload_bytes
    specs = [('precopy', (), owner_jobs['precopy'], (), 'send')]
    if details['branch'] == 'producer_consumer':
        # Producer-consumer (aiv_all_gather_mesh_1d.h:104-145): consumers poll
        # each peer's fragment flag via WaitFlag(targetRank) and pull each
        # positive fragment via GetGmIn(targetRank) — both are remote reads.
        flag_names = []
        copy_names = []
        for idx, frag_bytes in enumerate(_producer_fragment_bytes(selection, count)):
            flag_name = f'gather_ready_{idx}'
            copy_name = f'gather_copy_{idx}'
            specs.append((flag_name, ('precopy',), {}, _mesh(r, FLAG_BYTES), 'read'))
            specs.append((copy_name, (flag_name,), {}, _mesh(r, frag_bytes), 'read'))
            flag_names.append(flag_name)
            copy_names.append(copy_name)
        gather_after = tuple(flag_names) if flag_names else ('precopy',)
        specs.append(('gather', gather_after, owner_jobs['gather'], (), 'send'))
        tail = ('gather', *copy_names)
    else:
        # RunCtrlCore (aiv_all_gather_mesh_1d.h:168-184): readiness write,
        # remote payload read, ack write, and post-arrival ack wait are
        # causally separated phases per directed peer.
        peer_splits = _partition(r, details['copy_owner_count'])
        ready_owners = {
            owner: (Action('record', FLAG_BYTES),) * peers
            for owner, peers in enumerate(peer_splits)
        }
        copy_owners = {
            owner: tuple(
                action for _ in range(peers)
                for action in (Action('wait', FLAG_BYTES), Action('copy', n))
            )
            for owner, peers in enumerate(peer_splits)
        }
        ack_record_owners = {
            owner: (Action('record', FLAG_BYTES),) * peers
            for owner, peers in enumerate(peer_splits)
        }
        ack_wait_owners = {
            owner: (Action('wait', FLAG_BYTES),) * peers
            for owner, peers in enumerate(peer_splits)
        }
        specs.append(('gather_ready', ('precopy',), ready_owners, _mesh(r, FLAG_BYTES), 'send'))
        specs.append(('gather', ('gather_ready',), copy_owners, _mesh(r, n), 'read'))
        specs.append(('gather_ack', ('gather',), ack_record_owners, _mesh(r, FLAG_BYTES), 'send'))
        specs.append(('gather_ack_wait', ('gather_ack',), ack_wait_owners, (), 'send'))
        tail = ('gather_ack_wait',)
    if 'post_barrier' in owner_jobs:
        # BarrierAll records peer flags remotely before waiting on incoming
        # peer flags and clearing/resetting local barrier state.
        barrier_send = {}
        barrier_wait = {}
        for owner, peers in enumerate(_partition(r, count)):
            if peers:
                barrier_send[owner] = (Action('record', FLAG_BYTES),) * peers
                barrier_wait[owner] = tuple(
                    action for _ in range(peers)
                    for action in (Action('wait', FLAG_BYTES), Action('record', FLAG_BYTES))
                )
        specs.append(('post_barrier', tail, barrier_send, _mesh(r, FLAG_BYTES), 'send'))
        specs.append(('post_barrier_wait', ('post_barrier',), barrier_wait, (), 'send'))
    return tuple(specs)


def contract(selection, available, rank=0):
    owner_jobs, count, details = jobs(selection,available,rank)
    n = selection.payload_bytes
    phases, all_counts = [], {}
    for name, after, owners, exchange, direction in _phase_specs(owner_jobs, selection, count, details):
        demand, counts = {}, {}
        for actions in owners.values():
            for action in actions:
                if not action.size:
                    continue
                for role, value in action.work().items():
                    demand[role] = demand.get(role,0)+value
                counts[action.kind] = counts.get(action.kind,0)+1
        phases.append(HcclPhase(name,after,tuple(demand.items()),exchange=exchange,
            owner_domain=tuple(owner for owner,actions in owners.items() if actions),
            control_counts=tuple(sorted(counts.items())),
            native_direction=direction))
        for key,value in counts.items(): all_counts[key] = all_counts.get(key,0)+value
    all_counts['sync_all_block_calls'] = count*details['sync_all_calls_per_launched_block']
    return HcclPlan(selection,rank,tuple(phases),count,n,tuple(sorted(all_counts.items())),(
        'TopoMatchOneLevel_full_group_peer_access_declared; physical_levels_do_not_add_algorithm_rounds',
        'whole_phase_preload_barrier_replaces_per_fragment_peer_publication',
        'successful_flag_DMA_only; extra_poll_iterations_unknown',
        'flag_scalar_SetValue_GetValue_compare_and_SyncAll_have_no_declared_latency',
        'first_operator_BarrierForFirstOP_ClearGM_not_selected',
        'one_normal_kernel_executor_slice; no_superkernel_or_repeat_segmentation',
        'only_active_copy_queue_and_flags_resident; unused_Reduce64_queue_compilation_unknown',
    ))


def build(op, geometry):
    raw = dict(op.metadata)['hccl_phase']
    selection = HcclSelection(**raw['selection'])
    units = geometry.resource_units.get('vector',geometry.compute_units)
    owner_jobs,count,details = jobs(selection,units,raw.get('rank',0))
    phase_owners = {
        name: owners
        for name, _, owners, _, _ in _phase_specs(owner_jobs, selection, count, details)
    }
    plan = contract(selection,units,raw.get('rank',0))
    phase = next(p for p in plan.phases if p.name == raw['phase'])
    selected = phase_owners[phase.name]
    if geometry.buffers.get('ub',0) < UB_RESIDENCY:
        raise ModelingError('hccl_aiv_gather_requires_copy_queue_and_flags_UB')
    work = Work(resources=dict(phase.work))

    def program(context):
        owners=[]
        for owner, actions in selected.items():
            actions=tuple(action for action in actions if action.size)
            prefix=[0]
            for action in actions:
                prefix.append(prefix[-1]+(ceil(action.size/UB_BATCH) if action.kind=='copy' else 1))
            def tile(index, *, owner=owner, actions=actions, prefix=tuple(prefix)):
                action_index=bisect_right(prefix,index)-1
                action=actions[action_index]
                if action.kind=='copy':
                    size=min(UB_BATCH,action.size-(index-prefix[action_index])*UB_BATCH)
                    read=Work(resources={'hbm':size,'mte2_aiv':size})
                    write=Work(resources={'hbm':size,'mte3':size})
                    return (Stage('read',context.bind(read),placement=context.owned(owner,units,read.resources)),
                            Stage('write',context.bind(write),after=('read',),
                                  placement=context.owned(owner,units,write.resources)))
                demand=Work(resources=action.work())
                return (Stage(action.kind,context.bind(demand),
                              placement=context.owned(owner,units,demand.resources)),)
            if prefix[-1]:
                allocations=(() if GROUP_BUFFER in context.group_buffers else (
                    context.allocate(f'owner{owner}:copy','ub',UB_RESIDENCY,(owner,)),))
                owners.append(Stage(f'owner{owner}',program=PipelineProgram(prefix[-1],1,tile),
                                    acquire=allocations,free=allocations))
        return KernelProgram(tuple(owners))
    return OperatorPlan(work,program,{
        'source':SOURCE,'hccl_selection':asdict(selection),'hccl_phase':phase.name,
        'execution_class':'vector','launched_blocks':count,'source_schedule':details,
        'active_owner_domain':phase.owner_domain,'scratch_payload_bytes':plan.scratch_bytes,
        'control_counts':dict(phase.control_counts),'flag_bytes_per_successful_operation':FLAG_BYTES,
        'logical_payload_wire_per_rank':(selection.ranks-1)*selection.payload_bytes,
        'unpriced_scalar_counts':{'SetValue':dict(phase.control_counts).get('record',0),
                                 'GetValue_compare_branch':dict(phase.control_counts).get('wait',0)},
        'unmodeled_services':plan.unmodeled,
        'coverage':'explicit_source_owner_copy_and_successful_flag_program',
    })
