"""HC arch35 compact-profile lowering: exact source slots, FIFO and terminals."""
from collections import deque
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_hc_arch35 import (
    COMPACT_PRE_PROFILE, _msplit_template, _pre_template,
)
from mindie_sim.prediction.hc_arch35_compact import (
    finite_owner_template, finite_template_available, first_owner,
)
from mindie_sim.prediction.hc_arch35_program import (
    phase_program, pre_geometry, pre_program,
)
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelineTemplateProgram, RecipeBook, Stage,
)
from mindie_sim.prediction.structure import parse_case
from test_hc_arch35_source import _layer0_raw_case, changed_view, raw_case, sample


def compact_op(op):
    meta=dict(op.metadata)
    meta['fusion_boundary']={**meta['fusion_boundary'],'traffic_profile':COMPACT_PRE_PROFILE}
    return replace(op,metadata=tuple(meta.items()))


def test_two_chunk_hand_pipeline_keeps_13ms_and_readiness():
    from mindie_sim.prediction.engine import Engine,Resource
    from mindie_sim.prediction.program import RecipeContext
    from mindie_sim.prediction.hc_arch35_compact import capsule
    ctx=RecipeContext(None,{},dict(read='read',vector='vector',write='write'),
        dict(read=2,vector=2,write=2),2,resource_units=dict(read=2,vector=2,write=2))
    def s(name,role,amount,after=()):
        return Stage(name,{role:amount},after=after,placement={role:(1,)})
    stages=(s('r0','read',3),s('c0','vector',1,('r0',)),
        s('t0','vector',3,('c0',)),s('w0','write',2,('t0',)),
        s('r1','read',3,('r0',)),s('c1','vector',1,('r1','t0')),
        s('t1','vector',3,('c1',)),s('w1','write',2,('t1','w0')))
    for program in (KernelProgram(stages),KernelProgram(capsule(ctx,stages,'compact'))):
        engine=Engine([Resource(k,2,units=2) for k in ('read','vector','write')])
        ready=engine.submit('late',release=7)
        end=program.start(engine,'twochunks',ready)
        engine.advance(horizon=7);assert not any(engine.serviced.values())
        engine.advance(until=(end,))
        assert end.time==20
        assert engine.serviced=={'read':6,'vector':8,'write':4}


def execute(case,op,*,blocker=None):
    execution=Execution(case,CostModel(case),recipes=RecipeBook.from_mapping(case.runtime.kernels))
    op,recipe=execution.prepare(op,0)
    context=execution._context(op,0,None,recipe=recipe)
    engine=execution.engine;engine.trace=deque(maxlen=5000)
    ready=engine.submit('input_ready',release=7)
    if blocker:
        role,owner,work,*offset=blocker
        resource=context.resource_names[role]
        engine.submit('external', {resource:work},release=7+(offset[0] if offset else .00002),
            placement={} if role=='hbm' else context.owned(owner,context.resource_units[role],(role,)))
    complete=execution.lower_program(context,recipe).start(engine,'hc',ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,));span=complete.time-7
    engine.advance()
    assert not engine.summary()['live_tasks'] and not engine.summary()['live_programs']
    assert not any(engine.allocated.values())
    return span,engine.summary(),list(engine.trace),dict(op.metadata)['program_execution']


def prefill_op(tokens, hidden=None):
    case=parse_case(_layer0_raw_case())
    ops=target_graphs(case,scope='body',phase='prefill',tokens=tokens,
                      context=2048,sequences=1)[0]
    op=next(o for o in ops if o.layer==0 and
            dict(o.metadata).get('fusion_role')=='hc_pre')
    return (case,changed_view(op,hidden=hidden)) if hidden else (case,op)


def _signature(program):
    result=[]
    for stage in program.stages:
        result.append((stage.name,dict(stage.work),stage.after,dict(stage.placement),
                       stage.coupled))
        nested=getattr(stage,'program',None)
        if isinstance(nested,ParallelProgram):
            result.extend(tuple(_signature(child) for child in nested.programs))
        elif isinstance(nested,KernelProgram):
            result.extend(_signature(nested))
    return tuple(result)


def collect_work(program):
    total={}
    for stage in program.stages:
        for key,amount in stage.work.items():
            total[key]=total.get(key,0)+amount
        nested=getattr(stage,'program',None)
        if isinstance(nested,ParallelProgram):
            for child in nested.programs:
                for key,amount in collect_work(child).items():
                    total[key]=total.get(key,0)+amount
        elif isinstance(nested,KernelProgram):
            for key,amount in collect_work(nested).items():
                total[key]=total.get(key,0)+amount
    return total


def every_stage(program):
    if isinstance(program,PipelineTemplateProgram):
        template=program.template
        for phase in template.phases:
            yield from phase.stages
        yield from template.drain
        return
    for stage in program.stages:
        yield stage
        nested=getattr(stage,'program',None)
        if isinstance(nested,ParallelProgram):
            for child in nested.programs:
                yield from every_stage(child)
        elif isinstance(nested,KernelProgram):
            yield from every_stage(nested)
        elif isinstance(nested,PipelineTemplateProgram):
            template=nested.template
            for phase in template.phases:
                yield from phase.stages
            yield from template.drain


def _context_program(case,op):
    execution=Execution(case,CostModel(case),recipes=RecipeBook.from_mapping(case.runtime.kernels))
    prepared,recipe=execution.prepare(op,0)
    context=execution._context(prepared,0,None,recipe=recipe)
    view=dict(prepared.metadata)['fusion_tensor_view']
    g=pre_geometry(view,execution.stage_geometry[0,prepared.stage])
    if g.tiling_key==1001:
        first,work,parts=_msplit_template(g,view['sinkhorn_iters'])
        second=None
    else:
        first,second,work=_pre_template(g,view['sinkhorn_iters'])
        parts=None
    return execution,prepared,context,g,first,second,parts


def unbound_program(case,op,*,compact):
    execution,prepared,context,g,first,second,parts=_context_program(case,op)
    return pre_program(g,first,second,context,compact=compact,parts=parts),g,context,first


def owner_program(case,op,*,owner=0):
    execution,prepared,context,g,first,second,parts=_context_program(case,op)
    if g.tiling_key==1001:
        stage1=parts[owner][0];terminal='projection_local'
    else:
        stage1=first[owner];terminal='projection_out'
    program=first_owner(g,stage1,context,terminal=terminal)
    return program,context,execution,g,stage1


def rated_op(tokens,phase,**factors):
    raw=_layer0_raw_case()
    base=parse_case(deepcopy(raw))
    rates=dict(base.machine.dies[0].rates)
    raw['machine'].setdefault('device',{})['rates']={
        **rates, **{role:rates[role]*factor for role,factor in factors.items()}}
    case=parse_case(raw)
    ops=target_graphs(case,scope='body',phase=phase,tokens=tokens,
                      context=2048,sequences=1)[0]
    op=next(o for o in ops if o.layer==0 and
            dict(o.metadata).get('fusion_role')=='hc_pre')
    return case,op


def run_owner(case,op,*,owner=0,horizon=None,competitor=None):
    program,context,execution,g,stage1=owner_program(case,op,owner=owner)
    engine=execution.engine;engine.trace=deque(maxlen=100000)
    ready=engine.submit('ready',release=7)
    if competitor:
        role,work,unit,delay=competitor
        engine.submit('competitor',{context.resource_names[role]:work},
                      release=7+delay,
                      placement=context.owned(unit,context.resource_units[role],(role,)))
    complete=program.start(engine,'owner',ready)
    if horizon is None:
        engine.advance(until=(complete,));engine.advance()
    else:
        engine.advance(horizon=horizon)
    return list(engine.trace),engine,context,g


def _named(trace,name):
    return [record for record in trace if record['task']==f'owner:{name}']


def _task_service(engine,fragment):
    total=0.0
    for task in engine._tasks.values():
        if fragment in task.event.name:
            total+=sum(task.service.values())
    return total


def test_compact_lowering_is_exact_source_stage_graph():
    case,op=prefill_op(128)
    program,context,execution,g,stage1=owner_program(case,op)
    expected=phase_program(stage1,context)
    assert _signature(program)==_signature(expected)
    # Explicit source slot/FIFO edges required by the r2 P1.
    stages={stage.name:stage for stage in program.stages}
    chunks=[(p,c) for p in (0,1,2) for c in (0,1)]
    for sub in (0,1):
        def name(p,c,kind): return f'p{p}.v{sub}.c{c}.{kind}'
        for j in range(2,len(chunks)):
            p,c=chunks[j]
            pp,pc=chunks[j-2]
            assert name(pp,pc,'cast_rms') in stages[name(p,c,'read')].after
            assert name(pp,pc,'publish_l1') in stages[name(p,c,'transpose')].after
        for p in (2,):
            release=f'p{p-2}.k3.operands'
            assert release in stages[f'p{p}.v0.c0.publish_l1'].after
            assert release in stages[f'p{p}.v1.c0.publish_l1'].after
        last_vector=f'p2.v{sub}.c1.transpose'
        last_publish=f'p2.v{sub}.c1.publish_l1'
        assert stages[f'rms_out{sub}'].after==(last_vector,last_publish)
    for panel in (0,1,2):
        for micro in range(4):
            operand=f'p{panel}.k{micro}.operands'
            assert f'p{panel}.weight' in stages[operand].after
            assert f'p{panel}.v0.c1.publish_l1' in stages[operand].after
            assert f'p{panel}.v1.c1.publish_l1' in stages[operand].after
            assert f'p{panel}.k{micro}.cube' not in stages[operand].after
            assert operand in stages[f'p{panel}.k{micro}.cube'].after
            if micro>=2:
                assert f'p{panel}.k{micro-2}.cube' in stages[operand].after
    assert stages['projection_out'].after==('p2.k3.cube',)


@pytest.mark.parametrize('rows,hidden',((64,5120),(256,5120),
                                        (3,4128),(32,32768)))
def test_compact_profile_preserves_service_dependencies_and_buffers(rows,hidden):
    case,op=prefill_op(rows,hidden)
    detailed=execute(case,op);compact=execute(case,compact_op(op))
    assert compact[1]['serviced']==pytest.approx(detailed[1]['serviced'])
    assert compact[1]['peak_allocated_per_unit']==detailed[1]['peak_allocated_per_unit']
    # The finite template adds retire/begin events but repeats the same source
    # stages; it does not delete or average logical work.
    assert compact[1]['completed_tasks']>=detailed[1]['completed_tasks']
    assert compact[1]['completed_tasks']<=int(detailed[1]['completed_tasks']*1.1)+16
    assert compact[3]['compact_pipeline']['applied']
    assert compact[3]['compact_pipeline']['stage_merge']=='none_across_source_slot_boundaries'
    assert compact[3]['compact_pipeline']['finite_template']=='ordered_phases_key1000_stage1'
    assert compact[0]==pytest.approx(detailed[0],rel=2e-3,abs=1e-9)


def test_multi_l1_panels_publish_rms_before_projection_and_keep_release():
    case,op=prefill_op(8,32768)
    detailed=execute(case,op);compact=execute(case,compact_op(op))
    assert compact[3]['compact_pipeline']['applied']
    assert compact[3]['compact_pipeline']['finite_template']=='ordered_phases_key1000_stage1'
    assert compact[1]['serviced']==pytest.approx(detailed[1]['serviced'])
    assert compact[0]==pytest.approx(detailed[0],rel=2e-3,abs=1e-9)
    rms=[r['end_ms'] for r in compact[2] if ':rms_out1' in r['task']]
    projection=[r['end_ms'] for r in compact[2] if ':projection_out' in r['task']]
    assert rms and projection and min(rms)<max(projection)


@pytest.mark.parametrize('tokens',(4097,))
def test_key1001_stage1_exact_and_local_stage2_stays_detailed(tokens):
    case,op=prefill_op(tokens)
    detailed=execute(case,op);compact=execute(case,compact_op(op))
    assert compact[3]['compact_pipeline']['applied']
    assert compact[3]['compact_pipeline']['finite_template']=='exact_reference'
    assert compact[3]['compact_pipeline']['stage2']=='local_vector_unchanged'
    assert compact[1]['serviced']==pytest.approx(detailed[1]['serviced'])
    assert compact[1]['completed_tasks']==detailed[1]['completed_tasks']
    assert compact[0]==pytest.approx(detailed[0])
    det_program,g,_,_=unbound_program(case,op,compact=False)
    cmp_program,_,_,_=unbound_program(case,op,compact=True)
    assert _signature(cmp_program)==_signature(det_program)
    assert collect_work(cmp_program)==pytest.approx(collect_work(det_program))
    det_names=sorted(stage.name for stage in every_stage(det_program)
                     if stage.name.startswith('b'))
    cmp_names=sorted(stage.name for stage in every_stage(cmp_program)
                     if stage.name.startswith('b'))
    assert cmp_names==det_names
    assert g.m_tiles>=1 and g.k_parts==1


def test_each_owner_work_and_terminal_materialization():
    case,op=prefill_op(1024)
    execution,prepared,context,g,first,second,parts=_context_program(case,op)
    for owner in (0,g.m_dim*g.k_parts-1):
        program,_,_,_,stage1=owner_program(case,op,owner=owner)
        detailed=phase_program(stage1,context)
        assert _signature(program)==_signature(detailed)
        detailed_work={}
        for phase in stage1:
            for key,amount in context.bind(phase.work).items():
                detailed_work[key]=detailed_work.get(key,0)+amount
        detailed_work={key:amount for key,amount in detailed_work.items() if amount}
        assert collect_work(program)==pytest.approx(detailed_work)
        terminals={stage.name:stage for stage in program.stages
                   if stage.name in ('projection_out','rms_out0','rms_out1')}
        assert set(terminals)=={'projection_out','rms_out0','rms_out1'}
        for name in terminals:
            phase=next(phase for phase in stage1 if phase.name==name)
            assert dict(terminals[name].work)==pytest.approx(context.bind(phase.work))


def test_repeated_lowering_creates_fresh_events_not_cached_occurrence_state():
    case,op=prefill_op(1024)
    first,_,_,_=unbound_program(case,op,compact=True)
    second,_,_,_=unbound_program(case,op,compact=True)
    assert first is not second
    first_stages=tuple(every_stage(first));second_stages=tuple(every_stage(second))
    assert len(first_stages)==len(second_stages)
    for left,right in zip(first_stages,second_stages):
        assert left is not right
        assert left.name==right.name and left.after==right.after
        assert dict(left.work)==dict(right.work)
    assert {id(stage) for stage in first_stages}.isdisjoint(
        {id(stage) for stage in second_stages})


def test_compact_work_and_projection_diagnostics_are_not_zeroed():
    case,op=prefill_op(1024)
    det=operator_costs(case,(op,),input_ready_ms=7)['operators'][0]
    cmp=operator_costs(case,(compact_op(op),),input_ready_ms=7)['operators'][0]
    for key in ('hbm_bytes','matrix_flops','mte2_bytes','mte3_bytes','vector_ops'):
        assert cmp['operator'][key]==pytest.approx(det['operator'][key])
    det_meta=det['operator']['metadata']['program_execution']
    cmp_meta=cmp['operator']['metadata']['program_execution']
    assert cmp_meta['physical_projection_flops']==det_meta['physical_projection_flops']
    assert cmp_meta['compact_pipeline']['work_conserved']
    assert det['duration_ms']>0 and cmp['duration_ms']>0


def test_ordinary_frontend_selects_opt_in_profile_without_case_mutation():
    raw=_layer0_raw_case()
    raw['runtime']['kernels']['lowerings']['native_fusion']['options']['traffic_profile']['hc_pre']=COMPACT_PRE_PROFILE
    before=deepcopy(raw)
    ops=target_graphs(raw,scope='body',phase='decode',tokens=1,context=2048,sequences=8)[0]
    op=next(op for op in ops if dict(op.metadata).get('fusion_role')=='hc_pre')
    row=operator_costs(raw,(op,),input_ready_ms=7)['operators'][0]
    assert raw==before
    assert row['operator']['metadata']['program_execution']['compact_pipeline']['applied']
    assert not row['service_before_input'] and not row['live_tasks']


def test_ordinary_frontend_compacts_real_prefill_caller():
    raw=_layer0_raw_case()
    raw['runtime']['kernels']['lowerings']['native_fusion']['options']['traffic_profile']['hc_pre']=COMPACT_PRE_PROFILE
    before=deepcopy(raw)
    case=parse_case(deepcopy(raw))
    ops=target_graphs(case,scope='body',phase='prefill',tokens=1024,
                      context=2048,sequences=1)[0]
    op=next(op for op in ops if dict(op.metadata).get('fusion_role')=='hc_pre')
    row=operator_costs(case,(op,),input_ready_ms=7)['operators'][0]
    assert raw==before
    assert row['operator']['metadata']['program_execution']['compact_pipeline']['applied']
    assert row['duration_ms']>0 and not row['live_tasks']
    assert not any(row['remaining_buffers'].values())


def test_slow_vector_window_preserves_input_ring_and_rms_fifo():
    case,op=rated_op(128,'prefill',vector=0.02)
    trace,engine,context,g=run_owner(case,op,horizon=7.005)
    assert engine.serviced.get(context.resource_names['mte2_aiv'],0)==pytest.approx(65536)
    assert engine.serviced.get(context.resource_names['mte3'],0)==0
    assert _task_service(engine,'rms_out')==0
    read_tasks=[record for record in trace if '.read' in record['task']]
    assert read_tasks and all(record['start_ms']>=7 for record in read_tasks)
    # Only the first two chunk reads per sub were admitted; no later panel read
    # service appears in the window.
    assert not [record for record in trace if record['task'].endswith('.read')
                and '.p1.' in record['task']]


def test_slow_mte3_window_waits_ub_release_and_rms_write_fifo():
    case,op=rated_op(128,'prefill',mte3=0.02)
    trace,engine,context,g=run_owner(case,op,horizon=7.005)
    assert _task_service(engine,'rms_out')==0
    stages={stage.name:stage for stage in owner_program(case,op)[0].stages}
    assert 'p0.v0.c0.publish_l1' in stages['p1.v0.c0.transpose'].after
    assert 'p0.v1.c0.publish_l1' in stages['p1.v1.c0.transpose'].after
    assert 'p2.v0.c1.publish_l1' in stages['rms_out0'].after
    assert 'p2.v1.c1.publish_l1' in stages['rms_out1'].after


def test_aiv_reader_competitor_matches_source_completion():
    case,op=rated_op(128,'prefill',vector=0.02)
    competitor=(0.001,16384,0)  # delay ms, bytes, AIV unit 0
    trace,engine,context,g=run_owner(case,op,horizon=7.01,
                                     competitor=('mte2_aiv',16384,0,0.001))
    record=next(record for record in trace if record['task']=='competitor')
    assert record['end_ms']-record['start_ms']==pytest.approx(0.00015515151515151515)
    assert sum(record['serviced'].values())==pytest.approx(16384)


def test_slow_mte1_release_closes_panel_reuse_without_waiting_cube():
    case,op=rated_op(128,'prefill',mte1=0.02)
    _program,context,_execution,_g,_stage1=owner_program(case,op)
    trace,engine,context,g=run_owner(
        case,op,competitor=('cube',0.05/context.resource_units['cube'],0,0))
    release=max(record['end_ms']-7 for record in trace
                if record['task'].startswith('owner:p0.k')
                and record['task'].endswith('.operands'))
    last_cube=max(record['end_ms']-7 for record in trace
                  if record['task'].startswith('owner:p0.k')
                  and record['task'].endswith('.cube'))
    for name in ('p2.weight','p2.v0.c0.publish_l1','p2.v1.c0.publish_l1'):
        starts=[record['start_ms']-7 for record in trace
                if record['task']==f'owner:{name}']
        assert starts==[pytest.approx(release)]
    assert release<last_cube


def test_slow_aiv_cannot_launch_mte1_or_cube_before_publication():
    case,op=rated_op(8,'decode',mte2_aiv=0.02)
    _program,context,_execution,_g,_stage1=owner_program(case,op)
    trace,engine,context,g=run_owner(
        case,op,competitor=('cube',0.0005/context.resource_units['cube'],0,0))
    read_ends=[record['end_ms']-7 for record in trace
               if record['task'].startswith('owner:p0.v') and record['task'].endswith('.read')]
    weight_ends=[record['end_ms']-7 for record in trace
                 if record['task']=='owner:p0.weight']
    operands=[record for record in trace
              if record['task'].startswith('owner:p0.k') and record['task'].endswith('.operands')]
    cubes=[record for record in trace
           if record['task'].startswith('owner:p0.k') and record['task'].endswith('.cube')]
    assert read_ends and weight_ends and max(weight_ends)<min(read_ends)
    assert operands and cubes
    assert min(record['start_ms']-7 for record in operands)>=max(read_ends)
    assert min(record['start_ms']-7 for record in cubes)>=max(read_ends)
    competitor=[record for record in trace if record['task']=='competitor']
    assert competitor[0]['end_ms']-competitor[0]['start_ms']==pytest.approx(0.0005)


def test_panel_two_reuse_waits_mte1_release_not_cube_completion():
    case,op=rated_op(128,'prefill',mte1=0.02)
    _program,context,_execution,_g,_stage1=owner_program(case,op)
    trace,engine,context,g=run_owner(
        case,op,competitor=('cube',0.05/context.resource_units['cube'],0,0))
    release=max(record['end_ms']-7 for record in trace
                if record['task'].startswith('owner:p0.k')
                and record['task'].endswith('.operands'))
    last_cube=max(record['end_ms']-7 for record in trace
                  if record['task'].startswith('owner:p0.k')
                  and record['task'].endswith('.cube'))
    weight_start=next(record['start_ms']-7 for record in trace
                      if record['task']=='owner:p2.weight')
    publish_starts=[record['start_ms']-7 for record in trace
                    if record['task'] in ('owner:p2.v0.c0.publish_l1',
                                          'owner:p2.v1.c0.publish_l1')]
    assert weight_start==pytest.approx(release)
    assert all(start==pytest.approx(release) for start in publish_starts)
    assert weight_start<last_cube
    competitor=[record for record in trace if record['task']=='competitor']
    assert competitor and weight_start<competitor[0]['end_ms']-7


@pytest.mark.parametrize('role',('hbm','vector'))
def test_rate_scan_keeps_same_work_and_source_dependencies(role):
    baseline,op=prefill_op(128);rates=dict(baseline.machine.dies[0].rates)
    spans=[]
    for factor in (.5,2):
        raw=deepcopy(_layer0_raw_case());raw['machine'].setdefault('device',{})['rates']={role:rates[role]*factor}
        case=parse_case(raw)
        detailed=execute(case,op);compact=execute(case,compact_op(op))
        spans.append(compact[0])
        assert compact[1]['serviced']==pytest.approx(detailed[1]['serviced'])
        assert compact[0]==pytest.approx(detailed[0],rel=2e-3,abs=1e-9)
    assert spans[0]>=spans[1]-1e-9


def run_template_owner(case,op,*,owner=0,horizon=None,competitor=None):
    program,context,execution,g,stage1=owner_program(case,op,owner=owner)
    terminal='projection_local' if g.tiling_key==1001 else 'projection_out'
    template=finite_owner_template(g,stage1,context,terminal=terminal)
    engine=execution.engine;engine.trace=deque(maxlen=100000)
    ready=engine.submit('ready',release=7)
    if competitor:
        role,work,unit,delay=competitor
        engine.submit('competitor',{context.resource_names[role]:work},
                      release=7+delay,
                      placement=context.owned(unit,context.resource_units[role],(role,)))
    complete=PipelineTemplateProgram(template).start(engine,'owner',ready)
    if horizon is None:
        engine.advance(until=(complete,));engine.advance()
    else:
        engine.advance(horizon=horizon)
    return list(engine.trace),engine,context,g,template


def test_finite_template_construction_is_bounded_by_classes_not_panels():
    assert finite_template_available()
    metrics={}
    for tokens in (64,128,256,1024,4096):
        case,op=prefill_op(tokens)
        _program,context,execution,g,stage1=owner_program(case,op)
        template=finite_owner_template(g,stage1,context)
        metrics[tokens]=template
        assert template.phase_objects<=2
        assert template.stage_objects<=110
        assert template.static_phase_classes==template.phase_objects
        assert template.tile_factory_calls==template.phase_objects
        assert sum(phase.count for phase in template.phases)==template.tiles
        assert template.dynamic_instances>=template.tiles
        assert all(phase.count>=1 and phase.end is None for phase in template.phases)
    assert metrics[1024].stage_objects==metrics[4096].stage_objects
    assert metrics[1024].dynamic_instances==20
    assert metrics[4096].dynamic_instances==80
    assert [p.count for p in metrics[1024].phases]==[1,9]
    assert [p.count for p in metrics[4096].phases]==[1,39]
    # Fresh occurrence: the same static classes are rebuilt, never an Engine
    # event/tail or a previous template's Stage object.
    case,op=prefill_op(1024)
    _p,context,execution,g,stage1=owner_program(case,op)
    left=finite_owner_template(g,stage1,context)
    right=finite_owner_template(g,stage1,context)
    assert left is not right
    for lphase,rphase in zip(left.phases,right.phases):
        assert lphase.stages[0] is not rphase.stages[0]


@pytest.mark.parametrize('tokens,owner',((64,0),(128,0),(1024,0),(1024,31)))
def test_finite_template_executes_same_source_work_and_events(tokens,owner):
    case,op=prefill_op(tokens)
    exact=run_owner(case,op,owner=owner)
    finite=run_template_owner(case,op,owner=owner)
    assert finite[1].serviced==pytest.approx(exact[1].serviced)
    assert finite[1].summary()['live_tasks']==0
    assert finite[1].summary()['live_programs']==0
    exact_end=max(record['end_ms'] for record in exact[0])
    finite_end=max(record['end_ms'] for record in finite[0])
    assert finite_end==pytest.approx(exact_end,rel=2e-3,abs=1e-9)
    assert not any(finite[1].allocated.values())
    assert finite[4].phase_objects<=2


def test_finite_template_preserves_window_and_competitor_controls():
    case,op=rated_op(128,'prefill',vector=0.02)
    exact=run_owner(case,op,horizon=7.005)
    finite=run_template_owner(case,op,horizon=7.005)
    for engine,context in ((exact[1],exact[2]),(finite[1],finite[2])):
        assert engine.serviced.get(context.resource_names['mte2_aiv'],0)==pytest.approx(65536)
        assert engine.serviced.get(context.resource_names['mte3'],0)==0
        assert _task_service(engine,'rms_out')==0
    case,op=rated_op(128,'prefill',mte3=0.02)
    exact=run_owner(case,op,horizon=7.005)
    finite=run_template_owner(case,op,horizon=7.005)
    assert _task_service(exact[1],'rms_out')==0
    assert _task_service(finite[1],'rms_out')==0
    case,op=rated_op(128,'prefill',vector=0.02)
    exact=run_owner(case,op,horizon=7.01,
                    competitor=('mte2_aiv',16384,0,0.001))
    finite=run_template_owner(case,op,horizon=7.01,
                              competitor=('mte2_aiv',16384,0,0.001))
    for trace in (exact[0],finite[0]):
        record=next(record for record in trace if record['task']=='competitor')
        assert record['end_ms']-record['start_ms']==pytest.approx(0.00015515151515151515)
