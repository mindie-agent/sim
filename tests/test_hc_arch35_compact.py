"""Finite HC source compression with real resource, output and buffer controls."""
from collections import deque
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_hc_arch35 import COMPACT_PRE_PROFILE
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.operator_analysis import target_graphs,operator_costs
from test_hc_arch35_source import sample,raw_case,changed_view


def compact_op(op):
    meta=dict(op.metadata)
    meta['fusion_boundary']={**meta['fusion_boundary'],'traffic_profile':COMPACT_PRE_PROFILE}
    return replace(op,metadata=tuple(meta.items()))


def test_two_chunk_hand_pipeline_keeps_13ms_and_readiness():
    from mindie_sim.prediction.engine import Engine,Resource
    from mindie_sim.prediction.program import KernelProgram,RecipeContext,Stage
    from mindie_sim.prediction.hc_arch35_compact import capsule
    ctx=RecipeContext(None,{},dict(read='read',vector='vector',write='write'),
        dict(read=2,vector=2,write=2),2,resource_units=dict(read=2,vector=2,write=2))
    def s(name,role,amount,after=()):
        return Stage(name,{role:amount},after=after,placement={role:(1,)})
    stages=(s('r0','read',3),s('c0','vector',1,('r0',)),
        s('t0','vector',3,('c0',)),s('w0','write',2,('t0',)),
        s('r1','read',3,('r0',)),s('c1','vector',1,('r1','t0')),
        s('t1','vector',3,('c1',)),s('w1','write',2,('t1','w0')))
    # read0 0..3, vector0 3..7, read1 3..6, vector1 7..11;
    # stores7..9 and11..13. Idle owner0 cannot lend capacity.
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


@pytest.mark.parametrize('rows,hidden',((1,5120),(3,4128),(8,5120),(16,5120),(32,5120)))
def test_same_work_buffers_and_published_rms_before_projection(rows,hidden):
    case,op,_=sample(rows);op=changed_view(op,hidden=hidden)
    detailed=execute(case,op);compact=execute(case,compact_op(op))
    assert compact[1]['serviced']==pytest.approx(detailed[1]['serviced'])
    assert compact[1]['peak_allocated_per_unit']==detailed[1]['peak_allocated_per_unit']
    assert compact[1]['completed_tasks']<detailed[1]['completed_tasks']*.8
    assert compact[3]['compact_pipeline']['applied']
    # First-stage RMS remains a producer-side branch; it is not moved after
    # the final Cube/projection path or second-stage full-kernel barrier.
    rms=[r['end_ms'] for r in compact[2] if ':rms_out1' in r['task']]
    projection=[r['end_ms'] for r in compact[2] if ':projection_out' in r['task']]
    assert rms and projection and min(rms)<max(projection)
    assert abs(compact[0]/detailed[0]-1)<.15


def test_multi_l1_panels_keep_detailed_buffer_release_program_visible():
    case,op,_=sample(8);op=changed_view(op,hidden=32768)
    detailed=execute(case,op);compact=execute(case,compact_op(op))
    assert compact[0]==detailed[0]
    assert compact[1]==detailed[1]
    assert not compact[3]['compact_pipeline']['applied']


def test_ordinary_frontend_selects_opt_in_profile_without_case_mutation():
    raw=raw_case()
    raw['runtime']['kernels']['lowerings']['native_fusion']['options']['traffic_profile']['hc_pre']=COMPACT_PRE_PROFILE
    before=deepcopy(raw)
    ops=target_graphs(raw,scope='body',phase='decode',tokens=1,context=2048,sequences=8)[0]
    op=next(op for op in ops if dict(op.metadata).get('fusion_role')=='hc_pre')
    row=operator_costs(raw,(op,),input_ready_ms=7)['operators'][0]
    assert raw==before
    assert row['operator']['metadata']['program_execution']['compact_pipeline']['applied']
    assert not row['service_before_input'] and not row['live_tasks']


@pytest.mark.parametrize('role',('hbm','mte2_aiv','mte1','vector','mte3'))
def test_source_profile_rate_scan_keeps_work_and_finite_outputs(role):
    baseline,op,_=sample(16);rates=dict(baseline.machine.dies[0].rates)
    spans=[]
    for factor in (.5,1,2):
        raw=deepcopy(raw_case());raw['machine'].setdefault('device',{})['rates']={role:rates[role]*factor}
        case=parse_case(raw)
        detailed=execute(case,op);compact=execute(case,compact_op(op))
        spans.append(compact[0])
        assert compact[1]['serviced']==pytest.approx(detailed[1]['serviced'])
        assert abs(compact[0]/detailed[0]-1)<.15
    assert spans[0]>=spans[1]>=spans[2]
