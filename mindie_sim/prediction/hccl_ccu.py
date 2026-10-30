"""Explicit CCU Mesh1D source phases without importing selector timing guesses.

[Role] Separate CCU SRAM, local memory traffic and physical peer transport.
[Invariants] CCU_MS one-shot reduction reads peers directly into CCU buffers;
  those arrivals are not HBM writes. CCU_SCHED AllGather writes peers' outputs.
  Engines and CCU ISA versions are explicit. No Cube/AIV or heuristic rate alias.
[Timeouts & Locks] Finite phase DAGs and eight-slot local-copy pipeline; no locks.
[Side-Effect Policy] Existing Engine programs only, without runtime/source I/O.
"""
from dataclasses import asdict
from math import ceil

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, HcclPlan, HcclSelection, SOURCE, SOURCE_REVISION
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, PipelineProgram, Stage

RECIPE = 'hccl_ccu_source_phase'
OPTIONAL = frozenset(('ccu_copy', 'ccu_buffer_bytes', 'ccu_reduce_elements', 'ccu_control'))
# These are source geometry, not measured throughput or source selector costs.
REDUCE_FRAGMENT = 4096
COPY_FRAGMENT = 8 * 4096
COPY_SLOTS = 8


def _phase(name, after=(), *, exchange=(), counts=(), **work):
    return HcclPhase(name, after, tuple((k,v) for k,v in work.items() if v),
                     exchange=exchange, control_counts=counts)


def _mesh(ranks, size):
    return tuple(tuple(0 if i==j else size for j in range(ranks)) for i in range(ranks))


def _kind(s):
    valid = {
        ('CCU_MS','allreduce','mesh_oneshot_v1'): ('oneshot',1),
        ('CCU_MS','allreduce','mesh_oneshot_v2'): ('oneshot',2),
        ('CCU_SCHED','allgather','mesh_mem2mem_v1'): ('gather',1),
        ('CCU_SCHED','allgather','mesh_mem2mem_v2'): ('gather',2),
    }
    key = s.expansion,s.collective,s.template
    if (key not in valid or s.topology != 'MESH_1D' or s.topology_levels != 1
            or s.source_revision != SOURCE_REVISION or s.buffer_mode != 'separate'):
        raise ModelingError('hccl_ccu_requires_explicit_engine_ISA_template_and_separate_single_mesh')
    kind, version = valid[key]
    if s.ranks < 2:
        raise ModelingError('hccl_ccu_mesh_requires_at_least_two_channel_connected_ranks')
    if kind == 'oneshot' and (s.ranks > 8 or s.payload_bytes > REDUCE_FRAGMENT or s.dtype_bytes not in (2,4)):
        raise ModelingError('hccl_ccu_oneshot_single_fragment_requires_R2to8_N4096_nonexpanding_2or4byte_SUM')
    return kind, version


def contract(selection, rank=0):
    """Source-compatible phase contract, not evidence of deployed selection.

    OneShot is a single 4096-byte-or-smaller GroupReduce fragment. AllGather
    declares one equal-payload executor repeat; its local GroupCopy retains
    the source's eight 32KiB slots. Input and output do not alias.
    """
    s = selection
    kind, version = _kind(s)
    if not 0 <= rank < s.ranks:
        raise ModelingError('hccl_ccu_rank_outside_group')
    r,n = s.ranks,s.payload_bytes
    if not n:
        return HcclPlan(s,rank,(_phase('empty'),),1,0,unmodeled=('zero_payload_outer_executor_no_kernel',))
    pre = _phase('pre_sync', ccu_control=3*(r-1), counts=(
        ('write_variable_notify',2*(r-1)),('notify_wait',r-1)))
    phases=[pre]
    if kind=='oneshot':
        phases.extend((
            _phase('local_input_to_ccu',('pre_sync',),hbm=n,ccu_copy=n,ccu_buffer_bytes=n,
                   counts=(('local_copy',1),)),
            _phase('remote_input_to_ccu',('pre_sync',),exchange=_mesh(r,n),
                   hbm=(r-1)*n,ccu_buffer_bytes=(r-1)*n,counts=(('remote_read',r-1),)),
            _phase('load_wait',('local_input_to_ccu','remote_input_to_ccu'),ccu_control=1,
                   counts=(('event_wait',1),)),
            _phase('local_reduce',('load_wait',),ccu_reduce_elements=(r-1)*(n//s.dtype_bytes),
                   ccu_buffer_bytes=(r+1)*n,counts=(('local_reduce',1),)),
            _phase('reduce_wait',('local_reduce',),ccu_control=1,counts=(('event_wait',1),)),
            _phase('ccu_to_output',('reduce_wait',),hbm=n,ccu_copy=n,ccu_buffer_bytes=n,
                   counts=(('local_copy',1),)),
            _phase('output_wait',('ccu_to_output',),ccu_control=1,counts=(('event_wait',1),)),
        ))
        final=('output_wait',)
        # Template CalcScratchMultiple asks for R*N despite the narrow kernel
        # not using HBM scratch. Preserve capacity request, never charge traffic.
        scratch=r*n
    else:
        chunks=ceil(n/COPY_FRAGMENT)
        phases.extend((
            _phase('peer_write',('pre_sync',),exchange=_mesh(r,n),hbm=2*(r-1)*n,
                   counts=(('remote_write',r-1),('self_event_record',1)),ccu_control=1),
            _phase('local_copy',('pre_sync',),hbm=2*n,ccu_copy=2*n,ccu_buffer_bytes=2*n,
                   ccu_control=2*chunks,counts=(('local_copy',2*chunks),('event_wait',2*chunks))),
            _phase('write_wait',('peer_write','local_copy'),ccu_control=ceil(r/16),
                   counts=(('event_wait',ceil(r/16)),)),
        ))
        final=('write_wait',)
        scratch=0
    phases.append(_phase('post_sync',final,ccu_control=2*(r-1),
                         counts=(('notify_record',r-1),('notify_wait',r-1))))
    counts={}
    for phase in phases:
        for key,value in phase.control_counts:counts[key]=counts.get(key,0)+value
    counts['ccu_version']=version
    return HcclPlan(s,rank,tuple(phases),1,scratch,tuple(sorted(counts.items())),(
        'unpriced_CCU_services_are_not_zero_latency_evidence',
        'CCU_SRAM_port_work_is_logical_buffer_bytes_not_bank_transaction_amplification',
        'control_packet_bytes_task_submission_and_kernel_launch_not_inferred',
        'peer_transport_uses_whole_phase_readiness_not_submessage_progress',
        'CCU_buffer_handle_allocation_not_charged_as_NPU_UB_or_HBM',
        'one_executor_repeat; no_inplace_or_two_die_topology',
    ))


def buffer_geometry(selection):
    """Source resource handles and active finite-slot domain, not HBM capacity."""
    kind,version=_kind(selection)
    if kind=='oneshot':
        return {'space':'CCU SRAM','buffer_handle_bytes':4096,
                'allocated_buffer_handles':128*8,'allocated_event_handles':128*(3 if version==2 else 1),
                'active_single_fragment_handles':selection.ranks,
                'active_payload_bytes':selection.ranks*selection.payload_bytes}
    return {'space':'CCU SRAM','buffer_handle_bytes':4096,
            'allocated_buffer_handles':8*8,'allocated_event_handles':8*(2 if version==2 else 1),
            'copy_slots':COPY_SLOTS,'bytes_per_copy_slot':COPY_FRAGMENT}


def build(op, geometry):
    raw=dict(op.metadata)['hccl_phase']
    selection=HcclSelection(**raw['selection'])
    plan=contract(selection,raw.get('rank',0))
    phase=next(p for p in plan.phases if p.name==raw['phase'])
    omitted={k:v for k,v in phase.work if v and k in OPTIONAL and k not in geometry.resource_units}
    def demand(values):
        return Work(resources={k:v for k,v in values if k not in omitted and v})
    work=demand(phase.work)
    def lower(ctx):
        if selection.collective!='allgather' or phase.name!='local_copy':
            return KernelProgram((Stage('service',ctx.bind(work)),))
        n=selection.payload_bytes
        full_groups,remainder=divmod(n,COPY_SLOTS*COPY_FRAGMENT)
        tail_full,tail=divmod(remainder,COPY_FRAGMENT)
        def copies(count,size):
            def tile(index):
                value=size(index)
                payload=demand((('hbm',value),('ccu_copy',value),('ccu_buffer_bytes',value)))
                wait=demand((('ccu_control',1),))
                return (Stage('read',ctx.bind(payload)),
                        Stage('read_wait',ctx.bind(wait),after=('read',)),
                        Stage('write',ctx.bind(payload),after=('read_wait',)),
                        Stage('write_wait',ctx.bind(wait),after=('write',)))
            return PipelineProgram(count,COPY_SLOTS,tile)
        stages=[]
        if full_groups:
            stages.append(Stage('full_groups',program=copies(full_groups*COPY_SLOTS,lambda _:COPY_FRAGMENT)))
        if remainder:
            # CalGoSize's remainder fragment precedes the n whole fragments.
            stages.append(Stage('tail',after=('full_groups',) if full_groups else (),
                                program=copies(tail_full+bool(tail),
                                    lambda i:tail if tail and i==0 else COPY_FRAGMENT)))
        return KernelProgram(tuple(stages))
    return OperatorPlan(work,lower,{
        'source':SOURCE,'hccl_selection':asdict(selection),'hccl_phase':phase.name,
        'source_execution_domain':'CCU; neither_AIC_nor_AIV',
        'source_buffer_geometry':buffer_geometry(selection),
        'scratch_capacity_request_bytes':plan.scratch_bytes,
        'control_counts':dict(phase.control_counts),'unpriced_service_work':omitted,
        'service_units':{'ccu_copy':'bytes moved by CCU LocalCopy primitive',
                         'ccu_buffer_bytes':'logical CCU SRAM bytes read plus written',
                         'ccu_reduce_elements':'binary SUM element operations',
                         'ccu_control':'explicit notification/event-wait API calls'},
        'coverage':'source_HBM_lower_bound_with_unpriced_CCU_services' if omitted else 'source_declared_CCU_services',
        'unmodeled_services':plan.unmodeled,
    })
