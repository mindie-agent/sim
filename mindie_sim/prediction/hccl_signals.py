"""Remote HCCL notification events, including payload-free propagation.

[Role] Carry explicit source notify edges through ordinary topology routes.
[Invariants] A local wait has no independent network delay. Each remote signal
  traverses each hop once after both endpoints are ready. Unknown frame bytes
  stay zero, without skipping propagation or inventing a one-byte payload.
[Timeouts & Locks] O(explicit notification edges) virtual tasks; no I/O or locks.
[Side-Effect Policy] Submits events to the caller's Engine only.
"""
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from .collectives import _ready_by_rank, _joint_complete
from .communication import MessageFlow
from .program import KernelProgram, Stage

RECIPE = 'hccl_remote_notify'


@dataclass(frozen=True)
class SignalGroup:
    ranks: tuple[int, ...]
    topology: object
    endpoints: tuple[tuple[int, int], ...]
    signal_bytes: int = 0

    def start_participants(self, engine, name, ready_by_rank, *, owner=''):
        initial = _ready_by_rank(self.ranks, ready_by_rank)
        events = {rank: [initial[rank]] for rank in self.ranks}
        for index, (src_index, dst_index) in enumerate(self.endpoints):
            src, dst = self.ranks[src_index], self.ranks[dst_index]
            prefix = f'{name}:signal{index}:r{src}-r{dst}'
            prior = engine.submit(prefix + ':ready', after=(initial[src], initial[dst]),
                owner=owner, metadata={'source_rank':src, 'destination_rank':dst,
                                      'transport_schedule':'remote_notify'})
            route = self.topology.route_message(MessageFlow(src, dst, self.signal_bytes))
            for hop_index, hop in enumerate(route.hops):
                # RoutedHop.demand can share Tx/Rx and a switch capacity while
                # wire accounting remains a single physical message.
                service = engine.submit(f'{prefix}:hop{hop_index}:service', hop.demand,
                                        after=(prior,), owner=owner, coupled=True)
                prior = engine.submit(f'{prefix}:hop{hop_index}:arrival',
                    after=(service,), latency=hop.latency_ms, owner=owner)
            events[src].append(prior)
            events[dst].append(prior)
        return {rank:engine.submit(f'{name}:r{rank}:complete', after=tuple(after), owner=owner)
                for rank, after in events.items()}


@dataclass(frozen=True)
class SignalPlan:
    groups: tuple[SignalGroup, ...]

    @property
    def ranks(self):
        return tuple(sorted(r for g in self.groups for r in g.ranks))

    def start(self, engine, name, ready, *, owner=''):
        ends = self.start_participants(engine, name, {r:ready for r in self.ranks}, owner=owner)
        return _joint_complete(engine, name, ends.values(), ready, owner)

    def start_participants(self, engine, name, ready_by_rank, *, owner=''):
        _ready_by_rank(self.ranks, ready_by_rank)
        result = {}
        for index, group in enumerate(self.groups):
            result.update(group.start_participants(engine, f'{name}:group{index}',
                {r:ready_by_rank[r] for r in group.ranks}, owner=owner))
        return result


@dataclass(frozen=True)
class RemoteNotify:
    # Existing topology binds physical group ranks even for zero-volume groups.
    algorithm = 'direct_peer'

    def lower(self, context):
        if context.collective_plan is None:
            raise ModelingError('hccl_remote_notify_requires_bound_topology')
        if any(value for role, value in context.native_work.items() if role != 'link'):
            raise ModelingError('hccl_remote_notify_requires_transport_only_work')
        spec = dict(context.op.metadata)['hccl_remote_signal']
        groups = context.collective_plan().groups
        plan = SignalPlan(tuple(SignalGroup(g.ranks, g.topology,
            tuple(map(tuple, spec['endpoints'])), spec.get('bytes') or 0) for g in groups))
        return KernelProgram((Stage('notify', program=plan),), minimum_ms=context.minimum_ms)
