"""Owner-local source access counts and serialized accumulation on real Engine."""
from collections import deque
from copy import deepcopy

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from test_moe_init_mxfp8 import case as init_case,graph,RECIPE
from test_grouped_dequant_arch35 import ordinary,selected


def services(raw,**rates):
    raw=deepcopy(raw)
    raw['machine'].setdefault('device',{})['compute_arch']={'instruction_services':{
        role:{'unit':'calls','owner_axis':'vector','issue_ns':ns,
              'provenance':{'kind':'synthetic_capacity_control_not_measurement'}}
        for role,ns in rates.items()}}
    return raw


def changed(case,**intervals):
    raw=case.to_dict()
    for device in raw['machine']['devices']:
        device['compute_arch']={'instruction_services':{
            role:{'unit':'calls','owner_axis':'vector','issue_ns':ns,
                'provenance':{'kind':'synthetic_capacity_control_not_measurement'}}
            for role,ns in intervals.items()}}
    return parse_case(raw)


def execute(case,op):
    case=parse_case(case)
    execution=Execution(case,CostModel(case),recipes=RecipeBook.from_mapping(case.runtime.kernels))
    op,recipe=execution.prepare(op,0)
    context=execution._context(op,0,None,recipe=recipe)
    engine=execution.engine;engine.trace=deque(maxlen=12000)
    ready=engine.submit('late',release=7)
    end=execution.lower_program(context,recipe).start(engine,'source',ready)
    engine.advance(horizon=7);assert not any(engine.serviced.values())
    engine.advance(until=(end,))
    assert not engine.summary()['live_tasks'] and not engine.summary()['live_programs']
    assert not any(engine.allocated.values())
    return end.time-7,engine,op


def test_count_mode_known_source_scalar_services_remain_visible_when_unpriced():
    raw=init_case(16);op=next(o for o in graph(raw,16) if o.recipe==RECIPE)
    _,engine,prepared=execute(raw,op)
    view=dict(prepared.metadata)['program_execution']['primitive_services']
    assert view['unpriced_service_work']=={'scalar_ub_access':289,'scalar_alu_issue':288,'pipe_barrier':1}
    assert not view['priced_service_work']
    assert not any('scalar_count_sum' in row['task'] for row in engine.trace)


@pytest.mark.parametrize('factor',(.5,1,2))
def test_count_mode_load_add_chain_uses_owner0_not_all_idle_scalar_units(factor):
    raw=init_case(16);op=next(o for o in graph(raw,16) if o.recipe==RECIPE)
    raw['runtime']['kernels']['invocations']=[{'match':{},'startup_ms':0,'minimum_ms':0}]
    before,old,_=execute(raw,op)
    priced=services(raw,scalar_ub_access=1000/factor,scalar_alu_issue=2000/factor,pipe_barrier=3000/factor)
    after,engine,_=execute(priced,op)
    assert after-before==pytest.approx((289*.001+288*.002+.003)/factor)
    assert engine.serviced['scalar_ub_access@r0:s0']==pytest.approx(289)
    assert engine.serviced['scalar_alu_issue@r0:s0']==pytest.approx(288)
    assert engine.serviced['pipe_barrier@r0:s0']==pytest.approx(1)
    assert {key:engine.serviced[key] for key in old.serviced}==pytest.approx(old.serviced)
    trace={row['task']:row for row in engine.trace}
    for index in (0,1,287):
        load=next(row for name,row in trace.items() if name.endswith(f':tile{index}:get_count'))
        add=next(row for name,row in trace.items() if name.endswith(f':tile{index}:add_count'))
        assert load['end_ms']<=add['start_ms']
        assert load['placement']['scalar_ub_access@r0:s0']==(0,)
        assert add['placement']['scalar_alu_issue@r0:s0']==(0,)
        if index<287:
            nxt=next(row for name,row in trace.items() if name.endswith(f':tile{index+1}:get_count'))
            assert add['end_ms']<=nxt['start_ms']


@pytest.mark.parametrize('rows',((8,)*64,(0,2,0,1)))
def test_dsq_scalar_scan_visits_zero_groups_and_idle_owners_once(rows):
    base,op=selected(rows)
    old_span,old,_=execute(base,op)
    priced=changed(base,scalar_gm_access=1000)
    span,engine,prepared=execute(priced,op)
    diag=dict(prepared.metadata)['program_execution']['grouped_dequant_schedule']
    calls=diag['launched_aiv']*len(rows)
    assert engine.serviced['scalar_gm_access@r0:s0']==pytest.approx(calls)
    assert {key:engine.serviced[key] for key in old.serviced}==pytest.approx(old.serviced)
    assert span>old_span
    assert span>=len(rows)*.001
    per_owner={}
    for event in engine.trace:
        amount=event['serviced'].get('scalar_gm_access@r0:s0',0)
        if amount:
            owner=event['placement']['scalar_gm_access@r0:s0']
            assert len(owner)==1
            per_owner[owner[0]]=per_owner.get(owner[0],0)+amount
    assert per_owner==pytest.approx(dict.fromkeys(range(diag['launched_aiv']),len(rows)))
    # Every first data group waits for its actual descriptor prefix.
    names={row['task']:row for row in engine.trace}
    for name,event in names.items():
        if name.endswith(':group_scale') and event['demand']:
            descriptor=names[name.rsplit(':',1)[0]+':group_descriptor']
            assert descriptor['end_ms']<=event['start_ms']


def test_declared_gm_interval_changes_service_without_repeating_native_quantization():
    base,op,_=ordinary(64,2048)
    spans=[];payloads=[]
    for factor in (.5,1,2):
        span,engine,_=execute(changed(base,scalar_gm_access=1000/factor),op)
        spans.append(span)
        payloads.append({key:value for key,value in engine.serviced.items() if 'scalar_gm_access' not in key})
    assert spans[0]>spans[1]>spans[2]
    assert payloads[0]==pytest.approx(payloads[1])
    assert payloads[1]==pytest.approx(payloads[2])
