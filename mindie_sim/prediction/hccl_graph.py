"""Explicit native or analytic expansion on the ordinary collective caller DAG.

[Role] Compose declared local phases and physical peer transport on one Engine.
[Invariants] Each original collective is one invocation, never one per phase.
  Local HBM and link traffic have different owners; both are counted once.
  Selecting a source template is explicit, not inferred from a model or SKU.
  Analytic compositions carry distinct coverage and never native identity.
[Timeouts & Locks] O(operators * ranks squared) metadata, no I/O or locks.
[Side-Effect Policy] Returns new ops/recipes without changing caller declarations.
"""
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .hccl_contracts import select
from . import hccl_aicpu, hccl_aiv, hccl_aiv_gather, hccl_ccu, hccl_ccu_two_die, hccl_analytic_hierarchy
from .collectives import PairwiseCollective
from .mechanism import PlannedRecipe, Work
from .hccl_signals import RemoteNotify, RECIPE as SIGNAL

NETWORK = 'hccl_source_peer_exchange'


class SourceMeshPeers(PairwiseCollective):
    algorithm = 'direct_peer'

    def plan_options(self, op):
        return {'max_inflight_per_rank': dict(op.metadata)['hccl_selection']['ranks'] - 1}

    def accounting_description(self, accounting):
        result = accounting.to_dict()
        serial = result.pop('setup_ms')
        return {**result, 'setup_ms': 0.0, 'unused_serial_round_estimate_ms': serial,
                'latency_model': 'concurrent_source_peer_channels; per_hop_arrival'}


def recipes(kernels):
    if not kernels.get('hccl') and not kernels.get('analytic_hierarchy'):
        return {}
    return {hccl_aicpu.RECIPE: PlannedRecipe(hccl_aicpu.build),
            hccl_aiv.RECIPE: PlannedRecipe(hccl_aiv.build),
            hccl_aiv_gather.RECIPE: PlannedRecipe(hccl_aiv_gather.build),
            hccl_ccu.RECIPE: PlannedRecipe(hccl_ccu.build),
            hccl_ccu_two_die.RECIPE: PlannedRecipe(hccl_ccu_two_die.build),
            hccl_analytic_hierarchy.RECIPE: PlannedRecipe(hccl_analytic_hierarchy.build),
            NETWORK: SourceMeshPeers(path_model='store_forward'), SIGNAL: RemoteNotify()}


def lower(ops, runtime, source_dtype):
    bindings = runtime.kernels.get('hccl',{})
    analytic = runtime.kernels.get('analytic_hierarchy', {})
    if not bindings and not analytic:
        return ops
    if set(bindings)-{'allreduce','allgather','reduce_scatter'}:
        raise ModelingError('hccl_caller_supports_allreduce_allgather_reduce_scatter')
    if set(analytic) - {'allreduce'}:
        raise ModelingError('analytic_hierarchy_caller_supports_allreduce')
    result = []
    for original in ops:
        meta = dict(original.metadata)
        axis = meta.get('group_axis','tp')
        raw = bindings.get(original.family)
        analytic_raw = analytic.get(original.family)
        source_selected = raw is not None and axis in raw.get('axes', ['tp'])
        is_analytic = analytic_raw is not None and axis in analytic_raw.get('axes', ['tp'])
        if source_selected and is_analytic:
            raise ModelingError('collective_cannot_select_both_native_and_analytic_hierarchy')
        if is_analytic:
            raw = analytic_raw
        if (not source_selected and not is_analytic or 'hccl_source_group' in meta
                or 'analytic_hierarchy_group' in meta):
            result.append(original)
            continue
        if original.source_boundary or meta.get('device_kernel_group'):
            raise ModelingError('hccl_cannot_replace_another_native_kernel_boundary')
        if axis not in {'tp','pcp','etp'}:
            raise ModelingError('hccl_source_caller_requires_uniform_intra_replica_group')
        r = getattr(runtime.parallel,axis)
        if r < 2:
            result.append(original)
            continue
        factors = {'allreduce':2*(r-1)/r,'allgather':r-1,'reduce_scatter':(r-1)/r}
        n = original.link_bytes / factors[original.family]
        if not math.isclose(n,round(n),rel_tol=0,abs_tol=1e-6):
            raise ModelingError('hccl_caller_requires_integer_input_payload')
        options = {k:v for k,v in raw.items() if k != 'axes'}
        options.setdefault('dtype_bytes',int(dtype_bytes(source_dtype)))
        selector = hccl_analytic_hierarchy.select if is_analytic else select
        selection = selector(options,collective=original.family,ranks=r,payload_bytes=round(n))
        modules = {'AIV': hccl_aiv, 'AICPU_TS': hccl_aicpu,
                   'CCU_MS': hccl_ccu, 'CCU_SCHED': hccl_ccu,
                   'ANALYTIC': hccl_analytic_hierarchy}
        if selection.expansion not in modules:
            raise ModelingError('hccl_source_expansion_not_implemented:' + selection.expansion)
        module = modules[selection.expansion]
        if selection.expansion == 'AIV' and selection.collective == 'allgather':
            module = hccl_aiv_gather
        if selection.template == hccl_ccu_two_die.TEMPLATE:
            module = hccl_ccu_two_die
        if selection.expansion == 'AIV' and selection.block_limit is None:
            raise ModelingError('hccl_aiv_caller_requires_explicit_block_limit')
        plan = (module.contract(selection,selection.block_limit)
                if selection.expansion == 'AIV' else module.contract(selection))
        group = original.op_id+('.analytic_hierarchy' if is_analytic else '.hccl')
        entry = group+'.entry'
        ids = {p.name:group+'.'+p.name for p in plan.phases}
        netids = {p.name:ids[p.name]+'.network' for p in plan.phases if p.exchange or p.signals}
        kernel = ('ANALYTIC_HIERARCHY_allreduce' if is_analytic else
                  f'HCCL_{selection.expansion}_{selection.collective}_{selection.template}')
        common = {**meta,('analytic_hierarchy_group' if is_analytic else 'hccl_source_group'):group,
                  'hccl_selection':selection.to_dict(),
                  'local_work_scope':'representative_rank0; transport_covers_full_group',
                  **({} if is_analytic else {'native_kind':kernel}), 'implementation':kernel,
                  'invocation_class':('analytic_collective' if is_analytic else
                                      'vector' if selection.expansion == 'AIV' else
                                      'ccu' if selection.expansion.startswith('CCU_') else 'aicpu'),
                  'coverage':('explicit_analytic_three_level_mesh_composition' if is_analytic else
                              'explicit_source_template_with_whole_phase_peer_barriers')}
        if selection.expansion == 'AIV' and selection.template == 'mesh_twoshot':
            # The pinned split puts extra elements in early consumer slices.
            # Rank zero has the largest local receive/reduce shard; transport
            # retains the exact unequal rows instead of repeating that shard.
            common['local_rank_model'] = 'rank0_largest_integer_shard'
        # Logical entry/exit share the original stream. Independent physical
        # phase streams allow source branches to overlap without host relaunch.
        def info(is_entry=False,is_exit=False):
            result = {'id':group,'entry':is_entry,'exit':is_exit,'kernel':kernel}
            if is_entry and selection.expansion == 'AIV':
                result['buffers'] = [{
                    'id': hccl_aiv.GROUP_BUFFER, 'pool': 'ub',
                    'bytes': hccl_aiv.UB_RESIDENCY,
                    'units': list(range(plan.thread_count)),
                }]
            return result
        def op(name,deps,metadata,**changes):
            return Work().apply(replace(original,op_id=name,deps=tuple(deps),
                metadata=tuple({**common,**metadata}.items()),**changes))
        result.append(op(entry,original.deps,{'device_kernel_group':info(True)},
                         family='hccl_boundary',recipe='hccl_entry',source_boundary=''))
        for phase in plan.phases:
            deps = tuple(x for parent in phase.after for x in
                         (ids[parent],*((netids[parent],) if parent in netids else ()))) or (entry,)
            result.append(op(ids[phase.name],deps,{
                'device_kernel_group':info(),
                'hccl_phase':{'selection':selection.to_dict(),'rank':0,'phase':phase.name},
                **({'invocation_class':'vector'} if selection.expansion == 'AIV' else {}),
            },family='hccl_local',recipe=module.RECIPE,source_boundary='kernel',
                stream=ids[phase.name]))
            if phase.exchange:
                network_meta = {**common,'device_kernel_group':info(),
                    'collective':'alltoallv','collective_id':netids[phase.name],
                    'steps':r-1,'peer_volume_basis':'group_mean',
                    'peer_traffic':{'kind':'matrix','payload_bytes':phase.exchange}}
                network_meta.pop('reverse_of_collective',None)
                wire = Work(resources={'link':sum(map(sum, phase.exchange))/r})
                result.append(wire.apply(replace(original,op_id=netids[phase.name],deps=deps,
                    family='alltoallv',recipe=NETWORK,source_boundary='',
                    stream=netids[phase.name],metadata=tuple(network_meta.items()))))
            elif phase.signals:
                signal_matrix = [[0] * r for _ in range(r)]
                for src, dst in phase.signals:
                    signal_matrix[src][dst] += selection.signal_bytes or 0
                signal_meta = {**common, 'device_kernel_group':info(),
                    'collective':'alltoallv', 'collective_id':netids[phase.name],
                    'steps':r-1, 'peer_volume_basis':'group_mean',
                    'peer_traffic':{'kind':'matrix','payload_bytes':signal_matrix},
                    'hccl_remote_signal':{
                        'endpoints':phase.signals, 'bytes':selection.signal_bytes,
                        'wire_coverage':('propagation_only_unknown_frame' if selection.signal_bytes is None
                                         else 'explicit_declared_signal_frame')},
                }
                signal_meta.pop('reverse_of_collective', None)
                wire = Work(resources={'link':sum(map(sum,signal_matrix))/r})
                result.append(wire.apply(replace(original, op_id=netids[phase.name], deps=deps,
                    family='hccl_signal', recipe=SIGNAL, source_boundary='',
                    stream=netids[phase.name], metadata=tuple(signal_meta.items()))))
        consumed = {p for phase in plan.phases for p in phase.after}
        tails = tuple(x for phase in plan.phases if phase.name not in consumed
                      for x in (ids[phase.name],*((netids[phase.name],) if phase.name in netids else ())))
        result.append(op(original.op_id,tails,{'device_kernel_group':info(False,True)},
                         family='hccl_boundary',recipe='hccl_exit',source_boundary=''))
    return result
