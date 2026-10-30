"""Two endpoint-die CCU_MS AllGather from the explicit source channel map.

[Role] Retain two CCU owners, shared input reads, broadcast and notification DAGs.
[Invariants] Endpoint dies do not partition NPU ranks or select physical links.
  Each owner loads the same input once; only the smaller peer group copies self.
  Remote arrival precedes local waits, which add issue work but no second RTT.
[Timeouts & Locks] Single <=4096-byte fragment, finite O(R squared) metadata.
[Side-Effect Policy] In-memory source contracts and existing Engine programs only.
"""
from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, HcclPlan, HcclSelection, SOURCE, SOURCE_REVISION
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, Stage

RECIPE = 'hccl_ccu_two_die_source_phase'
TEMPLATE = 'mesh_2die_broadcast_v1'
SERVICES = ('ccu_copy', 'ccu_buffer_bytes', 'ccu_control')
FRAGMENT_BYTES = 4096


def channels(selection):
    """Return each rank's two sorted peer sets and its self-copy owner."""
    s = selection
    if (s.expansion != 'CCU_MS' or s.collective != 'allgather' or s.template != TEMPLATE
            or s.topology != 'TWO_DIE_REGULAR' or s.topology_levels != 1
            or s.buffer_mode != 'separate' or s.source_revision != SOURCE_REVISION
            or s.ranks < 4 or s.ranks % 2 or s.ranks > 16
            or not 0 <= s.payload_bytes <= FRAGMENT_BYTES):
        raise ModelingError('hccl_ccu_two_die_requires_regular_R4to16_even_single_fragment_CCUMS_AllGather_v1')
    mapping = s.channel_die_map
    if len(mapping) != s.ranks or any(len(row) != s.ranks for row in mapping):
        raise ModelingError('hccl_ccu_two_die_requires_complete_source_channel_die_map')
    groups = []
    for rank, row in enumerate(mapping):
        if row[rank] is not None or any(
                isinstance(die, bool) or die not in (0, 1)
                for peer, die in enumerate(row) if peer != rank):
            raise ModelingError('hccl_ccu_channel_die_map_requires_null_self_and_peer_endpoint_0_or_1')
        peers = tuple(tuple(peer for peer, die in enumerate(row) if die == owner)
                      for owner in (0, 1))
        counts = tuple(map(len, peers))
        if not all(counts) or abs(counts[0] - counts[1]) != 1:
            raise ModelingError('hccl_ccu_two_die_regular_requires_nonempty_peer_counts_differing_by_one')
        groups.append((peers, 0 if counts[0] < counts[1] else 1))
    # One local shard stands for the full TP group. Equal outgoing and incoming
    # degrees per endpoint make each phase's local work truly representative.
    # Skew requires per-rank CCU execution, not a rank-zero timing shortcut.
    for owner in (0, 1):
        degrees = {len(peers[owner]) for peers, _ in groups}
        degrees.update(sum(mapping[src][dst] == owner for src in range(s.ranks))
                       for dst in range(s.ranks))
        if len(degrees) != 1:
            raise ModelingError('hccl_ccu_two_die_requires_homogeneous_endpoint_degrees_for_representative_rank')
    return tuple(groups)


def contract(selection, rank=0):
    s = selection
    groups = channels(s)
    if not 0 <= rank < s.ranks:
        raise ModelingError('hccl_ccu_rank_outside_group')
    if not s.payload_bytes:
        return HcclPlan(s, rank, (HcclPhase('empty'),), 0, 0)
    n = s.payload_bytes
    peers, self_owner = groups[rank]
    phases = [HcclPhase('executor_pre', work=(('hccl_control', 2),),
                       control_counts=(('local_thread_notify', 1), ('local_thread_wait', 1)))]

    def phase(name, after=(), *, owner=None, work=(), exchange=(), signals=(), counts=()):
        phases.append(HcclPhase(name, after, tuple((k, v) for k, v in work if v),
                               exchange=exchange, owner_domain=() if owner is None else (owner,),
                               signals=signals, control_counts=counts))

    edges = [tuple((src, dst) for src in range(s.ranks) for dst in groups[src][0][owner])
             for owner in (0, 1)]
    for owner in (0, 1):
        tag = f'die{owner}'
        count = len(peers[owner])
        phase(tag + '_pre_record', ('executor_pre',), owner=owner,
              work=(('ccu_control', 2 * count),), counts=(('write_variable_notify', 2 * count),))
        # Output address and token are distinct notification messages. Keeping
        # the two phases separate preserves each frame's transaction rounding.
        for field in ('output', 'token'):
            phase(tag + '_pre_' + field, (tag + '_pre_record',), signals=edges[owner])
    pre_arrivals = tuple(f'die{owner}_pre_{field}' for owner in (0, 1) for field in ('output', 'token'))
    for owner in (0, 1):
        tag = f'die{owner}'
        count = len(peers[owner])
        local = int(owner == self_owner)
        phase(tag + '_pre_wait', pre_arrivals, owner=owner,
              work=(('ccu_control', count),), counts=(('notify_wait', count),))
        phase(tag + '_input_to_ccu', (tag + '_pre_wait',), owner=owner,
              work=(('hbm', n), ('ccu_copy', n), ('ccu_buffer_bytes', n)),
              counts=(('local_copy', 1),))
        phase(tag + '_load_wait', (tag + '_input_to_ccu',), owner=owner,
              work=(('ccu_control', 1),), counts=(('event_wait', 1),))
        matrix = tuple(tuple(n if s.channel_die_map[src][dst] == owner else 0 for dst in range(s.ranks))
                       for src in range(s.ranks))
        received = sum(row[rank] for row in matrix)
        phase(tag + '_broadcast', (tag + '_load_wait',), owner=owner, exchange=matrix,
              work=(('hbm', received + local * n), ('ccu_copy', local * n),
                    ('ccu_buffer_bytes', (count + local) * n)),
              counts=(('remote_write', count), ('local_copy', local)))
        phase(tag + '_write_wait', (tag + '_broadcast',), owner=owner,
              work=(('ccu_control', 1),), counts=(('event_wait', 1),))
        phase(tag + '_post_record', (tag + '_write_wait',), owner=owner,
              work=(('ccu_control', count),), counts=(('notify_record', count),))
        phase(tag + '_post_signal', (tag + '_post_record',), signals=edges[owner])
    for owner in (0, 1):
        count = len(peers[owner])
        phase(f'die{owner}_post_wait', ('die0_post_signal', 'die1_post_signal'), owner=owner,
              work=(('ccu_control', count),), counts=(('notify_wait', count),))
    phase('executor_post', ('die0_post_wait', 'die1_post_wait'), work=(('hccl_control', 2),),
          counts=(('local_thread_notify', 1), ('local_thread_wait', 1)))
    counts = {}
    for p in phases:
        for key, value in p.control_counts:
            counts[key] = counts.get(key, 0) + value
    return HcclPlan(s, rank, tuple(phases), 2, 0, tuple(sorted(counts.items())), (
        'one_equal_payload_executor_repeat_single_CCU_4096B_or_smaller_fragment',
        'homogeneous_endpoint_degrees_only; no_per_rank_CCU_owner_skew',
        'whole_group_notify_and_peer_phase_barriers; no_submessage_overlap',
        'source_compatible_explicit_selection_not_captured_deployed_dispatch',
        'two_internal_CCU_kernel_launches_and_endpoint_address_setup_not_separately_priced',
        'CCU_SRAM_logical_bytes_not_bank_transactions_or_physical_handle_capacity',
        'unknown_signal_frame_is_propagation_only' if s.signal_bytes is None else 'explicit_signal_frame_bytes',
    ))


def build(op, geometry):
    raw = dict(op.metadata)['hccl_phase']
    selection = HcclSelection(**raw['selection'])
    plan = contract(selection, raw.get('rank', 0))
    phase = next(p for p in plan.phases if p.name == raw['phase'])
    if any(geometry.resource_units.get(role) != 2 for role in SERVICES):
        raise ModelingError('hccl_ccu_two_die_requires_two_explicit_units_for_copy_buffer_and_control')
    omitted = {role: value for role, value in phase.work
               if role == 'hccl_control' and role not in geometry.resource_units}
    work = Work(resources={role: value for role, value in phase.work if role not in omitted})

    def program(context):
        placement = {}
        for role in work.resources:
            if role in SERVICES:
                placement.update(context.placement(role, phase.owner_domain))
        return KernelProgram((Stage('service', context.bind(work), placement=placement),))

    return OperatorPlan(work, program, {
        'source': SOURCE, 'hccl_selection': selection.to_dict(), 'hccl_phase': phase.name,
        'source_execution_domain': 'two_CCU_endpoint_owners_not_AIC_or_AIV',
        'active_owner_domain': phase.owner_domain,
        'source_peer_groups': channels(selection)[plan.rank][0],
        'self_copy_owner': channels(selection)[plan.rank][1],
        'internal_ccu_kernel_count': 2, 'control_counts': dict(phase.control_counts),
        'source_buffer_geometry': {'space': 'CCU SRAM', 'buffer_handle_bytes': 4096,
            'allocated_handles_per_owner': 128 * 8, 'active_fragment_handles_per_owner': 1,
            'active_payload_bytes_per_owner': selection.payload_bytes},
        'unpriced_service_work': omitted, 'unmodeled_services': plan.unmodeled,
        'coverage': 'explicit_two_endpoint_CCU_MS_source_fragment_with_physical_notifications',
    })
