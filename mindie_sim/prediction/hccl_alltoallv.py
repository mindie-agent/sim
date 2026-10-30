"""Sparse source contract for single-channel AICPU Mesh1D AlltoAllV.

[Role] Compose source peer slots, scratch copies and physical transport on Engine.
[Invariants] Exact integer byte cells, including diagonal and empty peers, are
  caller inputs. At most 16 peer streams run concurrently. Slots, not rounds,
  serialize later peers. Read/write select different scratch copy directions.
  Each directed wire payload and each local HBM endpoint is charged once.
[Timeouts & Locks] O(ranks squared) cohort events; no locks or real-time waits.
[Side-Effect Policy] Pure source plans and existing Engine tasks; no I/O/device.
"""
from dataclasses import dataclass, replace
from math import ceil

from mindie_sim.schema.common import ModelingError
from .communication import MessageFlow
from .hccl_contracts import HcclPhase, HcclSelection, SOURCE, SOURCE_REVISION
from .mechanism import Work

SOURCE_FILES = {
    'template': 'src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc',
    'header': 'src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.h',
    'transfer': 'src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc',
}
OPTIONAL_SERVICES = frozenset(('hccl_copy', 'hccl_control'))


@dataclass(frozen=True)
class PeerExchange:
    name: str
    remote_rank: int
    send_bytes: int
    recv_bytes: int
    slot: int
    round: int
    dma_mode: str


@dataclass(frozen=True)
class AlltoAllVPlan:
    selection: HcclSelection
    rank: int
    phases: tuple[HcclPhase, ...]
    peers: tuple[PeerExchange, ...]
    thread_count: int
    scratch_bytes: int
    control_counts: tuple[tuple[str, int], ...]
    unmodeled: tuple[str, ...]


def peer_schedule(ranks, rank):
    """Exact CalcCommRankSetForOneLoop and CalcCclBuffIdx for one channel."""
    slots = min(16, ranks - 1)
    if not slots:
        return ()
    pairs = (slots + 1) // 2
    result = []
    for round_idx in range(ceil((ranks - 1) / slots)):
        remain = ranks - 1 - len(result)
        pair_count = (remain + 1) // 2 if remain < slots else pairs
        for distance in range(round_idx * pairs + 1, round_idx * pairs + pair_count + 1):
            left, right = (rank - distance) % ranks, (rank + distance) % ranks
            if left == right:
                result.append((left, 0, round_idx))
                break
            slot = (distance - 1) % pairs
            result.extend(((left, pairs - 1 - slot, round_idx),
                           (right, pairs + slot, round_idx)))
    return tuple(result)


def _copy(size):
    return (('hbm', 2 * size), ('hccl_copy', size)) if size else ()


def _control(name, after, records, waits):
    count = records + waits
    return HcclPhase(name, after, (('hccl_control', count),) if count else (),
                     control_counts=(('notify', records), ('wait', waits)))


def _matrix(selection, matrix, rank):
    r = selection.ranks
    values = tuple(tuple(row) for row in matrix)
    if (len(values) != r or any(len(row) != r for row in values)
            or any(type(v) is not int or v < 0 or v % selection.dtype_bytes
                   for row in values for v in row)):
        raise ModelingError('hccl_alltoallv_requires_exact_integer_element_byte_matrix')
    if not 0 <= rank < r or selection.payload_bytes != sum(values[rank]):
        raise ModelingError('hccl_alltoallv_local_input_must_equal_matrix_row_including_self')
    return values


def contract(selection, send_matrix, rank=0):
    """One rank's local DAG; peer IDs are dense communication-group ordinals.

    mesh_read is the explicitly declared PCIe Read branch; mesh_write is the
    explicitly declared non-PCIe Write branch. Both use one channel and separate
    user/scratch buffers. No hardware SKU or measurement selects either branch.
    """
    s = selection
    if (s.expansion != 'AICPU_TS' or s.collective != 'alltoallv'
            or s.template not in {'mesh_read', 'mesh_write'} or s.buffer_mode != 'separate'
            or s.topology != 'MESH_1D' or s.topology_levels != 1
            or s.source_revision != SOURCE_REVISION):
        raise ModelingError('hccl_alltoallv_requires_explicit_single_channel_separate_mesh_read_or_write')
    matrix = _matrix(s, send_matrix, rank)
    schedule = peer_schedule(s.ranks, rank)
    slots = min(16, s.ranks - 1)
    if not slots:
        return AlltoAllVPlan(s, rank, (HcclPhase('local_copy', work=_copy(matrix[rank][rank])),),
                            (), 1, 0, (), ('single_rank_local_copy_not_mesh_executor_dispatch',))
    read = s.template == 'mesh_read'
    phases = [_control('pre_sync', (), slots, slots)]
    first = [(peer, slot) for peer, slot, loop in schedule if loop == 0]
    start = 'pre_sync'
    if read:
        copies = []
        for peer, _ in first:
            name = f'precopy_{peer}'
            phases.append(HcclPhase(name, ('pre_sync',), _copy(matrix[rank][peer])))
            copies.append(name)
        # Source's first read loop joins all slave pre-copies before releasing
        # their transfers; the main thread then overlaps its diagonal copy.
        phases.extend((_control('first_precopy_join', tuple(copies), slots, slots),
                       _control('first_precopy_release', ('first_precopy_join',), slots, slots)))
        start = 'first_precopy_release'
    phases.append(HcclPhase('local_copy', (start,), _copy(matrix[rank][rank])))
    tails = {}
    peers = []
    for peer, slot, loop in schedule:
        previous = tails.get(slot, start)
        sent, received = matrix[rank][peer], matrix[peer][rank]
        prefix = f'peer_{min(rank, peer)}_{max(rank, peer)}'
        if read and loop:
            name = f'precopy_{peer}'
            phases.append(HcclPhase(name, (previous,), _copy(sent)))
            previous = name
        # One-way has one record and one wait total; duplex has two of each.
        ack_records = int(bool(sent if read else received))
        ack_waits = int(bool(received if read else sent))
        ack = prefix + '_ack'
        phases.append(_control(ack, (previous,), ack_records, ack_waits))
        phases.append(HcclPhase(prefix, (ack,), (('hbm', sent + received),) if sent + received else (),
                                owner_domain=(slot + 1,)))
        signal = prefix + '_signal'
        phases.append(_control(signal, (prefix,), ack_waits, ack_records))
        previous = signal
        if not read and received:
            previous = f'postcopy_{peer}'
            phases.append(HcclPhase(previous, (signal,), _copy(received)))
        tails[slot] = previous
        peers.append(PeerExchange(prefix, peer, sent, received, slot, loop,
                                  'read' if read else 'write'))
    phases.append(_control('post_sync', tuple(tails.values()) + ('local_copy',), slots, slots))
    # Actual allocator inputSliceStride/segmentation is caller/executor state.
    # This is a sufficient minimal common stride for this rank, not its measured
    # allocation. Slots remain reserved even when a particular peer is empty.
    extent = max((matrix[rank][p] if read else matrix[p][rank] for p, _, _ in schedule), default=0)
    counts = {}
    for phase in phases:
        for key, count in phase.control_counts:
            counts[key] = counts.get(key, 0) + count
    counts.update(peer_stream_slots=slots, peer_groups=ceil((s.ranks - 1)/slots),
                  directed_sends=sum(p.send_bytes > 0 for p in peers),
                  directed_receives=sum(p.recv_bytes > 0 for p in peers))
    return AlltoAllVPlan(s, rank, tuple(phases), tuple(peers), slots + 1, slots * extent,
                        tuple(sorted(counts.items())), (
        'multi_channel_port_group_split_and_remote_memory_branches_not_selected',
        'scratch_bytes_is_minimum_uniform_stride_capacity_not_executor_allocation',
        'executor_message_segmentation_and_transport_control_packet_bytes_unmodeled',
        'notify_wait_task_issue_unpriced_unless_hccl_control_service_declared',
        'network_and_endpoint_HBM_progress_use_whole_peer_phase_envelope',
    ))


def bind_local_work(values, context):
    """Bind local payload/copy/control work; retain absent physical services."""
    omitted = {role: value for role, value in values
               if value and role in OPTIONAL_SERVICES
               and role not in context.resource_names and role not in context.service_modes}
    work = Work(resources={role: value for role, value in values if role not in omitted})
    return context.bind(work), omitted


@dataclass(frozen=True)
class AlltoAllVGroup:
    """Standalone cohort adapter; ordinary frontend selection is caller-owned.

    bindings maps physical rank IDs to RecipeContext. Link work is routed by the
    existing topology; the local phases own HBM, copy and optional control work.
    Engine tasks carry no per-phase invocation floor or serial round estimate.
    """
    selection: HcclSelection
    send_matrix: tuple[tuple[int, ...], ...]
    ranks: tuple[int, ...]
    topology: object

    def plans(self):
        if len(self.ranks) != self.selection.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ModelingError('hccl_alltoallv_requires_distinct_complete_physical_ranks')
        return tuple(contract(replace(self.selection, payload_bytes=sum(self.send_matrix[rank])),
                              self.send_matrix, rank) for rank in range(len(self.ranks)))

    def start_participants(self, engine, name, ready_by_rank, bindings, *, owner=''):
        plans = self.plans()
        if set(ready_by_rank) != set(self.ranks) or set(bindings) != set(self.ranks):
            raise ModelingError('hccl_alltoallv_requires_ready_and_binding_for_each_rank')
        events = {(p.rank, phase.name): engine.future(f'{name}:rank{self.ranks[p.rank]}:{phase.name}')
                  for p in plans for phase in p.phases}
        prepared = {}
        peer_by_rank = {p.rank: {peer.name: peer for peer in p.peers} for p in plans}
        for plan in plans:
            rank = self.ranks[plan.rank]
            for phase in plan.phases:
                prefix = f'{name}:rank{rank}:{phase.name}'
                deps = tuple(events[plan.rank, key] for key in phase.after) or (ready_by_rank[rank],)
                if phase.name in peer_by_rank[plan.rank]:
                    prepared[plan.rank, phase.name] = engine.submit(prefix + ':ready', after=deps, owner=owner)
                    continue
                demand, omitted = bind_local_work(phase.work, bindings[rank])
                event = events[plan.rank, phase.name]
                engine.submit(prefix, demand, after=deps, owner=owner,
                              metadata={'hccl_source': SOURCE, 'unpriced_service_work': omitted},
                              on_complete=lambda _, target=event: engine.resolve(target))
        for plan in plans:
            for peer in plan.peers:
                if plan.rank > peer.remote_rank:
                    continue
                left, right = plan.rank, peer.remote_rank
                pair_name = peer.name
                deps = (prepared[left, pair_name], prepared[right, pair_name])
                if not peer.send_bytes and not peer.recv_bytes:
                    # RunSendRecv does nothing for a zero/zero pair: it cannot
                    # create a remote-ready barrier absent from the source.
                    for index in (left, right):
                        target = events[index, pair_name]
                        engine.submit(f'{name}:rank{self.ranks[index]}:{pair_name}:empty',
                                      after=(prepared[index, pair_name],), owner=owner,
                                      on_complete=lambda _, target=target: engine.resolve(target))
                    continue
                local = []
                for index in (left, right):
                    rank = self.ranks[index]
                    other = peer_by_rank[index][pair_name]
                    values = (('hbm', other.send_bytes + other.recv_bytes),)
                    demand, omitted = bind_local_work(values, bindings[rank])
                    local.append(engine.submit(f'{name}:rank{rank}:{pair_name}:endpoints', demand,
                                               after=deps, owner=owner,
                                               metadata={'unpriced_service_work': omitted}))
                wire = []
                for src, dst, size in ((left, right, peer.send_bytes), (right, left, peer.recv_bytes)):
                    if not size:
                        continue
                    source, destination = self.ranks[src], self.ranks[dst]
                    previous = engine.submit(f'{name}:{source}->{destination}:ready', after=deps, owner=owner)
                    route = self.topology.route_message(MessageFlow(source, destination, size))
                    for hop_index, hop in enumerate(route.hops):
                        service = engine.submit(f'{name}:{source}->{destination}:hop{hop_index}:service',
                                                hop.demand, after=(previous,), owner=owner, coupled=True)
                        previous = engine.submit(f'{name}:{source}->{destination}:hop{hop_index}:arrival',
                                                 after=(service,), latency=hop.latency_ms, owner=owner)
                    wire.append(previous)
                targets = (events[left, pair_name], events[right, pair_name])
                def complete(_, targets=targets):
                    for target in targets:
                        engine.resolve(target)
                engine.submit(f'{name}:{pair_name}:complete', after=tuple(local + wire),
                              owner=owner, on_complete=complete)
        return {self.ranks[p.rank]: events[p.rank, p.phases[-1].name] for p in plans}
